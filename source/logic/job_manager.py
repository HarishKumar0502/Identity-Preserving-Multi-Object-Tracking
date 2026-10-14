"""Video-processing job manager (the existing in-process job mechanism).

* Jobs run on a bounded ThreadPoolExecutor (``APP.max_concurrent_jobs``, default 1).
* Detector and ReID backend come from ``source/models/model_registry.py`` (loaded once
  per process and shared); the ReID backend is chosen automatically
  (``source/logic/inference_manager.py``): the client never picks a backend, device,
  precision or tracker parameter.
* Every job processes the whole video with ``VideoProcessor`` (``video_processor.py``)
  and the validated default tracker config.
* Job state is kept in memory and mirrored to ``outputs/<job_id>/job.json``; it is
  reloaded at start-up (``restore_jobs``), so job IDs survive a server restart until TTL.
* The uploaded input is deleted when its job ends; a failed job's partial outputs are
  removed. ``sweep_stale_files()`` (run at start-up) removes leftovers of earlier runs.
* ``shutdown()`` signals running jobs to stop and waits for the pool to finish.
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from source.config.settings import (APP, REID, TRACKER, AppConfig, TrackerConfig,
                                    baseline_tracker_config)
from source.models.model_registry import ModelRegistry

log = logging.getLogger("ps2.worker")


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    input_path: str
    output_dir: str
    params: dict
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    frames_processed: int = 0
    total_frames: int = 0
    error: Optional[str] = None
    summary: Optional[dict] = None
    # Set only after the written video has been validated; the container may
    # differ from the requested .mp4 when the encoder falls back (e.g. MJPG/.avi).
    output_path: Optional[str] = None
    output_codec: Optional[str] = None
    input_filename: Optional[str] = None     # original upload name (display only)
    reid_backend: Optional[dict] = None      # backend actually used (set when processing)
    processing_time_s: Optional[float] = None

    @property
    def requested_output(self) -> Path:
        return Path(self.output_dir) / "tracked.mp4"

    @property
    def output_video(self) -> Optional[Path]:
        return Path(self.output_path) if self.output_path else None

    @property
    def output_ready(self) -> bool:
        out = self.output_video
        return (self.status == JobStatus.COMPLETED and out is not None and out.is_file()
                and out.stat().st_size > 0)

    @property
    def mot_file(self) -> Path:
        return Path(self.output_dir) / "tracks_mot.txt"

    @property
    def progress(self) -> float:
        if self.status == JobStatus.COMPLETED:
            return 100.0
        if self.total_frames:
            return round(min(100.0, 100.0 * self.frames_processed / self.total_frames), 2)
        return 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["progress_percent"] = self.progress
        return d


def build_tracker_config(params: dict) -> TrackerConfig:
    base = baseline_tracker_config() if params.get("tracker_mode") == "baseline" else TRACKER
    overrides = {}
    if params.get("track_buffer") is not None:
        overrides["track_buffer"] = int(params["track_buffer"])
    if params.get("reassociation_threshold") is not None:
        overrides["reid_thresh"] = float(params["reassociation_threshold"])
    if params.get("high_confidence") is not None:
        overrides["high_conf"] = float(params["high_confidence"])
        overrides["new_track_conf"] = max(base.new_track_conf, float(params["high_confidence"]))
    return base.with_overrides(**overrides)


class JobManager:
    def __init__(self, app_cfg: AppConfig = APP, registry: Optional[ModelRegistry] = None):
        self.cfg = app_cfg
        self.registry = registry or ModelRegistry()
        self.jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=max(1, app_cfg.max_concurrent_jobs),
                                            thread_name_prefix="ps2-job")
        app_cfg.upload_dir.mkdir(parents=True, exist_ok=True)
        app_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ API
    @staticmethod
    def new_job_id() -> str:
        return uuid.uuid4().hex

    def pending_count(self) -> int:
        with self._lock:
            return sum(j.status in (JobStatus.QUEUED, JobStatus.PROCESSING) for j in self.jobs.values())

    def submit(self, job_id: str, input_path: Path, params: Optional[dict] = None,
               input_filename: Optional[str] = None) -> Job:
        out_dir = self.cfg.output_dir / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        job = Job(job_id=job_id, input_path=str(input_path), output_dir=str(out_dir),
                  params=dict(params or {}), input_filename=input_filename)
        with self._lock:
            self.jobs[job_id] = job
        self._persist(job)
        self._executor.submit(self._run, job)
        log.info("[job %s] queued (%s)", job_id, params)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self.jobs.get(job_id)

    def shutdown(self):
        self._cancel.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            for job in self.jobs.values():
                if job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
                    job.status, job.error = JobStatus.FAILED, "Server shut down before completion."
                    self._persist(job)
        self.registry.release()

    def cleanup_expired(self) -> int:
        """Delete uploads/outputs of finished jobs older than the TTL."""
        cutoff = time.time() - self.cfg.job_ttl_hours * 3600
        removed = 0
        with self._lock:
            stale = [j for j in self.jobs.values()
                     if j.status in (JobStatus.COMPLETED, JobStatus.FAILED)
                     and (j.finished_at or j.created_at) < cutoff]
            for j in stale:
                self.jobs.pop(j.job_id, None)
        for j in stale:
            shutil.rmtree(j.output_dir, ignore_errors=True)
            _unlink_quietly(Path(j.input_path))
            removed += 1
        return removed

    def sweep_stale_files(self) -> dict:
        """Remove leftovers of earlier server runs.

        Uploads with no live job and output folders older than the TTL, and spooled
        multipart temp files. Safe to call at start-up before any job is submitted.
        """
        with self._lock:
            live_inputs = {Path(j.input_path).resolve() for j in self.jobs.values()}
            live_outputs = {Path(j.output_dir).resolve() for j in self.jobs.values()}
        cutoff = time.time() - self.cfg.job_ttl_hours * 3600
        counts = {"uploads": 0, "temp": 0, "outputs": 0}
        for f in self.cfg.upload_dir.glob("*"):
            # Age-gated: another server instance sharing the folder may still own newer files.
            if (f.is_file() and f.resolve() not in live_inputs and f.stat().st_mtime < cutoff
                    and _unlink_quietly(f)):
                counts["uploads"] += 1
        temp_dir = getattr(self.cfg, "temp_dir", None)
        if temp_dir is not None and Path(temp_dir).is_dir():
            for f in Path(temp_dir).glob("*"):
                if f.is_file() and _unlink_quietly(f):
                    counts["temp"] += 1
        for d in self.cfg.output_dir.glob("*"):
            if (d.is_dir() and len(d.name) == 32 and d.resolve() not in live_outputs
                    and d.stat().st_mtime < cutoff):
                shutil.rmtree(d, ignore_errors=True)
                counts["outputs"] += 1
        if any(counts.values()):
            log.info("Removed stale files from earlier runs: %s", counts)
        return counts

    def restore_jobs(self) -> int:
        """Reload jobs of earlier server runs from ``outputs/<job_id>/job.json``.

        Finished jobs stay reachable (status, video, tracks, summary) across restarts until
        their TTL expires. Jobs that were still queued/processing when the previous server
        stopped are marked failed. Call at start-up, after ``sweep_stale_files``.
        """
        names = {f.name for f in fields(Job)}
        restored = 0
        for state_file in self.cfg.output_dir.glob("*/job.json"):
            job_dir = state_file.parent
            if len(job_dir.name) != 32 or job_dir.name in self.jobs:
                continue
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                job = Job(**{k: v for k, v in data.items() if k in names})
                job.status = JobStatus(job.status)
            except (OSError, ValueError, TypeError) as exc:
                log.warning("Skipping unreadable job state %s: %s", state_file, exc)
                continue
            if job.job_id != job_dir.name:
                continue
            job.output_dir = str(job_dir)  # the project may have moved since
            if job.output_path:
                job.output_path = str(job_dir / Path(job.output_path).name)
            if job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
                job.status, job.error = JobStatus.FAILED, "Server restarted before completion."
                job.finished_at = job.finished_at or time.time()
                self._remove_partial_outputs(job)
                self._persist(job)
            with self._lock:
                self.jobs[job.job_id] = job
            restored += 1
        if restored:
            log.info("Restored %d job(s) from earlier runs.", restored)
        return restored

    # -------------------------------------------------------------- execution
    def _persist(self, job: Job):
        try:
            path = Path(job.output_dir) / "job.json"
            path.write_text(json.dumps(job.to_dict(), indent=2, default=str), encoding="utf-8")
        except OSError as exc:  # pragma: no cover
            log.warning("[job %s] could not persist state: %s", job.job_id, exc)

    def _run(self, job: Job):
        from source.logic.video_processor import ProcessingCancelled, VideoError, VideoProcessor
        from source.models.detector.inference import DetectorError
        from source.models.reid.inference import ReIDModelError
        from source.utils.video_utils import VideoWriterError

        job.status, job.started_at = JobStatus.PROCESSING, time.time()
        self._persist(job)
        log.info("[job %s] processing started (%s)", job.job_id, job.input_filename or "upload")
        try:
            if self._cancel.is_set():
                raise ProcessingCancelled("Server is shutting down.")
            detector = self.registry.detector()
            backend = self.registry.reid()
            job.reid_backend = backend.info()
            tracker_cfg = build_tracker_config(job.params)
            processor = VideoProcessor(detector, backend, tracker_cfg, REID,
                                       inference_lock=self.registry.inference_lock)

            def progress(done: int, total: int):
                job.frames_processed, job.total_frames = done, total
                if done % 50 == 0:
                    self._persist(job)

            summary = processor.process(Path(job.input_path), job.requested_output, job.mot_file,
                                        progress_cb=progress, cancel_event=self._cancel,
                                        max_frames=self.cfg.max_frames)
            # The processor has already reopened and decoded the output; re-check
            # the files here so a job can never be "completed" without its results.
            out = Path(job.output_dir) / summary["output"]["filename"]
            if not out.is_file() or out.stat().st_size == 0:
                raise VideoWriterError("Output video is missing after processing.")
            if not job.mot_file.is_file():
                raise VideoWriterError("Tracking result file is missing after processing.")
            summary["tracker_config"] = {k: v for k, v in asdict(tracker_cfg).items()}
            summary["reid_backend_selection"] = getattr(self.registry, "selection", None)
            job.reid_backend = backend.info()  # may differ after a runtime fallback
            job.summary = summary
            job.frames_processed = summary["video"]["frame_count"]
            job.total_frames = max(job.total_frames, job.frames_processed)
            job.output_path, job.output_codec = str(out), summary["output"]["codec"]
            job.status = JobStatus.COMPLETED
            log.info("[job %s] completed: %d frames (%s), %d detections, %d IDs, %.2f FPS, "
                     "ReID %s, codec %s -> %s", job.job_id, summary["video"]["frame_count"],
                     summary["video"]["stop_reason"], summary["detection"]["total_detections"],
                     summary["tracking"]["num_unique_track_ids"],
                     summary["performance"]["average_processing_fps"] or 0,
                     job.reid_backend.get("backend"), job.output_codec, out)
        except (VideoError, VideoWriterError, ProcessingCancelled) as exc:
            job.status, job.error = JobStatus.FAILED, str(exc)
            log.warning("[job %s] failed: %s", job.job_id, exc)
        except (ReIDModelError, DetectorError) as exc:  # actionable model/runtime problems
            job.status = JobStatus.FAILED
            job.error = f"Model initialization failed: {exc}"
            log.error("[job %s] failed: %s", job.job_id, exc)
        except Exception as exc:  # noqa: BLE001 - report every failure on the job
            if "out of memory" in str(exc).lower():
                job.error = "Out of memory during inference; try a shorter or lower-resolution video."
                self.registry.release()
            else:  # details (paths, tracebacks) stay in the server log
                job.error = f"Internal processing error ({type(exc).__name__}); see server log."
            job.status = JobStatus.FAILED
            log.error("[job %s] failed with exception\n%s", job.job_id, traceback.format_exc())
        finally:
            job.finished_at = time.time()
            if job.started_at:
                job.processing_time_s = round(job.finished_at - job.started_at, 3)
            if job.status == JobStatus.FAILED:
                self._remove_partial_outputs(job)
            if not getattr(self.cfg, "keep_inputs", False):
                _unlink_quietly(Path(job.input_path))
            self._persist(job)

    @staticmethod
    def _remove_partial_outputs(job: Job):
        out_dir = Path(job.output_dir)
        for f in list(out_dir.glob("tracked.*")) + [job.mot_file]:
            _unlink_quietly(f)


def _unlink_quietly(path: Path, attempts: int = 5) -> bool:
    """Delete a file, retrying briefly: on Windows a scanner/indexer may hold it open."""
    for i in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return True
        except PermissionError:
            time.sleep(0.2 * (i + 1))
        except OSError as exc:
            log.warning("Could not delete %s: %s", path.name, exc)
            return False
    log.warning("Could not delete %s (file in use); it will be removed by the next sweep.",
                path.name)
    return False
