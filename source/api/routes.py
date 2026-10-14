"""HTTP routes: health/readiness, job submission, job status and results.

Every route reads the job manager (and its ``AppConfig``) from ``request.app.state.jobs``,
which ``source/main.py`` creates at start-up.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from source.config.settings import DETECTOR, REID
from source.logic.job_manager import JobManager, JobStatus
from source.utils.video_utils import MEDIA_TYPES

log = logging.getLogger("ps2.api")

router = APIRouter()

_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_CHUNK = 1 << 20


def _manager(request: Request) -> JobManager:
    return request.app.state.jobs


def _job_or_404(request: Request, job_id: str):
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="Unknown job ID.")
    job = _manager(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job ID.")
    return job


_JOB_ERRORS = {
    404: {"description": "Unknown job ID, or the output file is missing (e.g. expired)"},
    409: {"description": "Job is still queued/processing, or it failed (detail has the error)"},
}
_VIDEO_RESPONSES = {
    200: {"description": "The tracked (annotated) video as a file download. The container is "
                         "normally MP4 (H.264); AVI (MJPG) only if no MP4 encoder was usable.",
          "content": {"video/mp4": {"schema": {"type": "string", "format": "binary"}},
                      "video/x-msvideo": {"schema": {"type": "string", "format": "binary"}}}},
    **_JOB_ERRORS,
}
_TRACKS_RESPONSES = {
    200: {"description": "Tracks in MOTChallenge format: frame,id,x,y,w,h,score,-1,-1,-1",
          "content": {"text/plain": {"schema": {"type": "string", "format": "binary"}}}},
    **_JOB_ERRORS,
}


def _job_links(job_id: str) -> dict:
    return {"status_url": f"/jobs/{job_id}", "video_url": f"/jobs/{job_id}/video",
            "tracks_url": f"/jobs/{job_id}/tracks", "summary_url": f"/jobs/{job_id}/summary"}


@router.get("/health", tags=["system"])
def health(request: Request):
    from source.models.model_registry import available_backends
    from source.utils.device_utils import probe_cuda, resolve_device, tensorrt_version

    mgr = _manager(request)
    reg = mgr.registry.status()
    try:
        device = resolve_device(DETECTOR.device)
    except RuntimeError as exc:
        device = f"unavailable ({exc})"
    gpu = probe_cuda()
    return {
        "status": "ok" if reg["detector_loaded"] and reg["reid_loaded"] else "degraded",
        "detector": {"available": reg["detector_loaded"], "weights": Path(DETECTOR.weights).name,
                     "device": device, "error": reg["errors"].get("detector")},
        "reid": {"loaded": reg["reid_loaded"], "backend": reg["reid_backend"],
                 "selection": reg["reid_selection"], "checkpoint": Path(REID.checkpoint).name,
                 "embedding_dim": REID.embedding_dim, "input_size_hw": list(REID.input_size),
                 "error": reg["errors"].get("reid")},
        "hardware": {"cuda_gpu": gpu.get("name") if gpu["available"] else None,
                     "cuda_note": None if gpu["available"] else gpu.get("reason"),
                     "tensorrt_version": tensorrt_version()},
        "installed_inference_backends": available_backends(),
        "artifacts": {"onnx_model_present": REID.onnx_path.is_file(),
                      "tensorrt_engine_present": REID.tensorrt_path.is_file()},
        "video_output": {"opencv_version": cv2.__version__,
                         "codec_candidates": mgr.cfg.video_codecs},
        "jobs_pending": mgr.pending_count(),
    }


@router.get("/ready", tags=["system"])
def ready(request: Request):
    """Readiness: 200 only when the detector and a verified ReID backend are loaded."""
    reg = _manager(request).registry.status()
    is_ready = reg["detector_loaded"] and reg["reid_loaded"]
    return JSONResponse(status_code=200 if is_ready else 503,
                        content={"ready": is_ready, "detector_loaded": reg["detector_loaded"],
                                 "reid_loaded": reg["reid_loaded"], "errors": reg["errors"]})


@router.post("/track", tags=["tracking"], status_code=202)
async def track(
    request: Request,
    video: UploadFile = File(..., description="Video file (.mp4, .avi, .mov, .mkv, .m4v, .webm). "
                                              "The whole video is processed."),
):
    """Submit a video for tracking. Only the file is required; everything else is automatic."""
    mgr = _manager(request)
    cfg = mgr.cfg
    ext = Path(video.filename or "").suffix.lower()
    if ext not in cfg.allowed_extensions:
        await video.close()
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{ext or 'none'}'. "
                                                    f"Allowed: {', '.join(cfg.allowed_extensions)}.")
    if mgr.pending_count() >= cfg.max_queued_jobs:
        await video.close()
        raise HTTPException(status_code=429, detail="Too many jobs in progress; retry later.")

    job_id = mgr.new_job_id()
    dest = cfg.upload_dir / f"{job_id}{ext}"
    limit = cfg.max_upload_mb * (1 << 20)
    size = 0
    try:
        with open(dest, "wb") as f:  # stream to disk; never hold the video in RAM
            while chunk := await video.read(_CHUNK):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413,
                                        detail=f"File exceeds {cfg.max_upload_mb} MB limit.")
                f.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except OSError as exc:
        dest.unlink(missing_ok=True)
        log.error("Upload could not be stored: %s", exc)
        raise HTTPException(status_code=507, detail="The server could not store the upload "
                                                    "(disk full or not writable).")
    finally:
        await video.close()
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    from source.logic.video_processor import VideoError, probe_video
    try:
        meta = await run_in_threadpool(probe_video, dest)
    except VideoError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid video: {exc}")

    mgr.cleanup_expired()
    job = mgr.submit(job_id, dest, input_filename=Path(video.filename or "").name)
    reid = mgr.registry.status()["reid_backend"]
    return {"job_id": job.job_id, "status": job.status.value, "video": meta,
            "reid_backend": reid, **_job_links(job.job_id)}


@router.get("/jobs/{job_id}", tags=["jobs"])
def job_status(request: Request, job_id: str):
    job = _job_or_404(request, job_id)
    body = {"job_id": job.job_id, "status": job.status.value, "input_filename": job.input_filename,
            "progress_percent": job.progress, "frames_processed": job.frames_processed,
            "total_frames": job.total_frames, "error": job.error,
            "reid_backend": job.reid_backend, "processing_time_s": job.processing_time_s,
            "output_available": job.output_ready,
            "output_filename": job.output_video.name if job.output_video else None,
            "output_codec": job.output_codec,
            "stop_reason": (job.summary or {}).get("video", {}).get("stop_reason"),
            "result": None}
    if job.status == JobStatus.COMPLETED and job.summary:
        s = job.summary
        body["result"] = {
            "frames": s["video"]["frame_count"], "video_duration_s": s["video"]["duration_s"],
            "unique_track_ids": s["tracking"]["unique_track_ids"],
            "num_unique_track_ids": s["tracking"]["num_unique_track_ids"],
            "no_ids_reason": s["tracking"]["no_ids_reason"],
            "average_processing_fps": s["performance"]["average_processing_fps"],
            "warnings": _result_warnings(s), **_job_links(job.job_id)}
    return body


@router.get("/jobs/{job_id}/video", tags=["jobs"], response_class=FileResponse,
            responses=_VIDEO_RESPONSES)
def job_video(request: Request, job_id: str):
    """Download the tracked video of a completed job (``Content-Disposition: attachment``)."""
    job = _job_or_404(request, job_id)
    _require_completed(job)
    if not job.output_ready:
        raise HTTPException(status_code=404, detail="Output video file not found (expired or "
                                                    "removed).")
    out = job.output_video
    return FileResponse(out, media_type=MEDIA_TYPES.get(out.suffix.lower(), "application/octet-stream"),
                        filename=f"tracked_{job.job_id[:8]}{out.suffix.lower()}")


@router.get("/jobs/{job_id}/tracks", tags=["jobs"], response_class=FileResponse,
            responses=_TRACKS_RESPONSES)
def job_tracks(request: Request, job_id: str):
    """Per-frame tracks in MOTChallenge format: frame,id,x,y,w,h,score,-1,-1,-1."""
    job = _job_or_404(request, job_id)
    _require_completed(job)
    if not job.mot_file.is_file():
        raise HTTPException(status_code=404, detail="Tracking result file not found (expired "
                                                    "or removed).")
    return FileResponse(job.mot_file, media_type="text/plain",
                        filename=f"tracks_{job.job_id[:8]}.txt")


@router.get("/jobs/{job_id}/summary", tags=["jobs"])
def job_summary(request: Request, job_id: str):
    job = _job_or_404(request, job_id)
    _require_completed(job)
    if job.summary is None:
        raise HTTPException(status_code=409, detail="Summary not ready.")
    return {"job_id": job.job_id, "processing_time_s": job.processing_time_s, **job.summary}


def _result_warnings(summary: dict) -> list:
    v, out = summary["video"], []
    if v.get("ended_before_reported_length"):
        out.append(f"Input ended after {v['frame_count']} decodable frames although the file "
                   f"reports {v['reported_frame_count']} (truncated or partially corrupt file?).")
    skipped = v.get("skipped_unreadable_frames", 0) + v.get("skipped_malformed_frames", 0)
    if skipped:
        out.append(f"{skipped} unreadable frame(s) were skipped.")
    events = (summary.get("performance", {}).get("reid_backend") or {}).get("fallback_events")
    if events:
        out.append(f"ReID backend fell back during processing: {events[-1]['from']} -> "
                   f"{events[-1]['to']}.")
    return out


def _require_completed(job):
    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=409, detail=f"Job failed: {job.error}")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=f"Job is {job.status.value}; result not ready.")
