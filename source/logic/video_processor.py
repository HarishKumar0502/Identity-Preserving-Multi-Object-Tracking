"""Frame-by-frame video tracking: detect -> embed -> associate -> render -> write.

Strictly online/causal: each frame is processed using only past state.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from source.config.settings import REID, TRACKER, ReIDConfig, TrackerConfig
from source.logic.tracking.reid_extractor import ReIDExtractor
from source.logic.tracking.track_manager import Tracker, TrackOutput
from source.models.detector.inference import Detections, PersonDetector
from source.utils.video_utils import create_video_writer, validate_video_output

log = logging.getLogger("ps2.video")


class VideoError(RuntimeError):
    pass


class ProcessingCancelled(RuntimeError):
    pass


def probe_video(path: Path) -> Dict:
    """Open the file with OpenCV and decode one frame; raise VideoError if unreadable."""
    path = Path(path)
    if not path.is_file():
        raise VideoError("Input video file not found.")
    if path.stat().st_size == 0:
        raise VideoError("Input video file is empty.")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("File could not be opened as a video (unsupported codec/container, "
                             "or not a video file).")
        try:
            ok, frame = cap.read()
        except cv2.error as exc:
            raise VideoError(f"Video could not be decoded: {exc}") from exc
        if not ok or frame is None:
            raise VideoError("Video contains no decodable frames.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        try:
            ok2, frame2 = cap.read()
        except cv2.error:
            ok2, frame2 = False, None
        if not ok2 or frame2 is None:  # e.g. a still image renamed to .mp4
            raise VideoError("File contains a single image or only one decodable frame; "
                             "tracking needs a video.")
        return {"fps": fps if 0 < fps < 1000 else 25.0, "frame_count": max(n, 0),
                "width": frame.shape[1], "height": frame.shape[0]}
    finally:
        cap.release()


def _color(track_id: int):
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(c) for c in rng.integers(64, 256, 3))


def render(frame: np.ndarray, outputs: List[TrackOutput], frame_idx: int, n_dets: int) -> np.ndarray:
    fh, fw = frame.shape[:2]
    for o in outputs:
        x1, y1, x2, y2 = (int(round(v)) for v in o.box)
        if not o.observed and not (0 <= (x1 + x2) / 2 < fw and 0 <= (y1 + y2) / 2 < fh):
            continue  # predicted position has left the field of view
        color = _color(o.track_id)
        if o.observed:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {o.track_id}" + (" [recovered]" if o.state == "reassociated" else "")
        else:  # predicted position while occluded
            for x in range(x1, x2, 12):
                cv2.line(frame, (x, y1), (min(x + 6, x2), y1), color, 1)
                cv2.line(frame, (x, y2), (min(x + 6, x2), y2), color, 1)
            for y in range(y1, y2, 12):
                cv2.line(frame, (x1, y), (x1, min(y + 6, y2)), color, 1)
                cv2.line(frame, (x2, y), (x2, min(y + 6, y2)), color, 1)
            label = f"ID {o.track_id} (occluded)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1, th + 4)
        lx = int(np.clip(x1, 0, max(fw - tw - 4, 0)))
        cv2.rectangle(frame, (lx, ty - th - 4), (lx + tw + 4, ty), color, -1)
        cv2.putText(frame, label, (lx + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)
    visible = sum(o.observed for o in outputs)
    cv2.putText(frame, f"frame {frame_idx}  tracks {visible}  dets {n_dets}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


@dataclass
class StageTimer:
    totals: Dict[str, float] = field(default_factory=dict)

    @contextlib.contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - t0


class VideoProcessor:
    def __init__(self, detector: PersonDetector, reid_backend, tracker_cfg: TrackerConfig = TRACKER,
                 reid_cfg: ReIDConfig = REID, inference_lock: Optional[threading.Lock] = None):
        self.detector = detector
        self.backend = reid_backend
        self.tracker_cfg = tracker_cfg
        self.reid_cfg = reid_cfg
        self.extractor = ReIDExtractor(reid_backend, reid_cfg, tracker_cfg)
        self.lock = inference_lock or contextlib.nullcontext()

    # Consecutive failed reads tolerated (before the container's reported frame
    # count is reached) while trying to skip a corrupt frame.
    max_read_retries = 5
    log_every = 50

    def process(self, input_path: Path, output_path: Path, mot_path: Optional[Path] = None,
                progress_cb: Optional[Callable[[int, int], None]] = None,
                cancel_event: Optional[threading.Event] = None,
                max_frames: Optional[int] = None) -> Dict:
        """Track every readable frame of ``input_path`` and write the rendered video.

        ``output_path``'s suffix may be changed by the encoder fallback; the file
        actually written is reported as ``summary["output"]["filename"]`` (same
        directory). Raises ``VideoError`` / ``VideoWriterError`` if no frame could be
        processed or the written output does not validate.
        """
        try:
            import psutil
            proc = psutil.Process()
        except ImportError:  # pragma: no cover
            proc = None
        meta = probe_video(input_path)
        size = (meta["width"], meta["height"])
        reported = meta["frame_count"]
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            cap.release()
            raise VideoError("File could not be opened as a video.")
        log.info("Input opened: %dx%d @ %.3f fps, container reports %s frames (backend %s)",
                 size[0], size[1], meta["fps"], reported or "unknown", cap.getBackendName())
        writer = None
        writer_info = None
        mot_file = None
        tracker = Tracker(self.tracker_cfg)
        cfg = self.tracker_cfg
        timer = StageTimer()
        ids_seen: set = set()
        frames_processed = 0
        frames_read = 0          # successful reads, including skipped malformed frames
        pending_failures = 0     # failed reads not (yet) followed by a good frame
        skipped_unreadable = 0
        skipped_malformed = 0
        stop_reason = "end_of_stream"
        det_total = det_high = det_new = frames_with_dets = det_max = 0
        peak_rss = 0
        latencies = []
        t_start = time.perf_counter()
        try:
            writer, writer_info = create_video_writer(output_path, meta["fps"], size)
            if mot_path is not None:
                mot_file = open(mot_path, "w", encoding="utf-8")
            total = reported
            if max_frames:
                total = min(total, max_frames) if total else max_frames
                if not reported or max_frames < reported:
                    log.info("max_frames=%d: processing at most %d of %s frames",
                             max_frames, max_frames, reported or "unknown")
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise ProcessingCancelled("Processing was interrupted.")
                if max_frames and frames_processed >= max_frames:
                    stop_reason = "max_frames"
                    break
                with timer("decode"):
                    try:
                        ok, frame = cap.read()
                    except cv2.error as exc:  # decoder error on a corrupt packet
                        log.warning("Decoder error at read #%d: %s", frames_read + 1, exc)
                        ok, frame = False, None
                if not ok or frame is None:
                    position = frames_read + pending_failures
                    if reported and position < reported and pending_failures < self.max_read_retries:
                        pending_failures += 1
                        log.warning("Frame read failed at position %d of %d; retrying",
                                    position + 1, reported)
                        continue
                    break
                frames_read += 1
                if pending_failures:
                    log.warning("Skipped %d unreadable frame(s) before frame read #%d",
                                pending_failures, frames_read)
                    skipped_unreadable += pending_failures
                    pending_failures = 0
                if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
                    skipped_malformed += 1
                    log.warning("Skipping malformed frame (shape %s) at read #%d",
                                frame.shape, frames_read)
                    continue
                if (frame.shape[1], frame.shape[0]) != size:
                    frame = cv2.resize(frame, size)
                frame_idx = frames_processed + 1
                t_frame = time.perf_counter()
                with self.lock:
                    with timer("detect"):
                        dets: Detections = self.detector.detect(frame, frame_idx)
                    with timer("reid"):
                        app = self.extractor(frame, dets.boxes, dets.scores,
                                             only=dets.scores >= cfg.high_conf)
                with timer("associate"):
                    outputs = tracker.update(dets.boxes, dets.scores, app.embeddings,
                                             app.valid, app.reliable)
                latencies.append(time.perf_counter() - t_frame)
                with timer("render"):
                    render(frame, outputs, frame_idx, len(dets))
                with timer("encode"):
                    writer.write(frame)
                frames_processed += 1

                n = len(dets)
                n_high = int(np.sum(dets.scores >= cfg.high_conf))
                det_total += n
                det_high += n_high
                det_new += int(np.sum(dets.scores >= cfg.new_track_conf))
                frames_with_dets += n > 0
                det_max = max(det_max, n)
                for o in outputs:
                    if not o.observed:
                        continue
                    ids_seen.add(o.track_id)
                    if mot_file:
                        x1, y1, x2, y2 = o.box
                        mot_file.write(f"{frame_idx},{o.track_id},{x1:.2f},{y1:.2f},"
                                       f"{x2 - x1:.2f},{y2 - y1:.2f},{o.score:.4f},-1,-1,-1\n")
                if log.isEnabledFor(logging.DEBUG) or frames_processed % self.log_every == 1:
                    states = tracker.state_counts()
                    log.info("frame %d: dets=%d (>=high_conf %d) active=%d tentative=%d "
                             "occluded=%d lost=%d visible_ids=%s", frame_idx, n, n_high,
                             states.get("confirmed", 0) + states.get("reassociated", 0),
                             states.get("tentative", 0), states.get("occluded", 0),
                             states.get("lost", 0),
                             sorted(o.track_id for o in outputs if o.observed))
                if proc is not None and frames_processed % 25 == 1:
                    peak_rss = max(peak_rss, proc.memory_info().rss)
                if progress_cb is not None:
                    progress_cb(frames_processed, total)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if mot_file is not None:
                mot_file.close()
        elapsed = time.perf_counter() - t_start
        if frames_processed == 0:
            raise VideoError("No frames could be decoded and processed.")
        if stop_reason == "end_of_stream" and reported and frames_read + pending_failures < reported:
            log.warning("Stream ended after %d readable frames; container reported %d",
                        frames_read, reported)
        with timer("validate"):
            decoded = validate_video_output(writer_info.path, expected_frames=frames_processed,
                                            size=size)
        out_size = writer_info.path.stat().st_size
        log.info("Output validated: %s (codec %s, %d frames decoded, %d bytes)",
                 writer_info.path.name, writer_info.codec, decoded, out_size)

        st = tracker.stats
        id_note = None
        if not ids_seen:
            if det_total == 0:
                id_note = ("No person detections in any processed frame: the input most likely "
                           "contains no detectable people (not a tracking failure).")
            elif det_new == 0:
                id_note = (f"People were detected, but none scored >= new_track_conf "
                           f"({cfg.new_track_conf}), so no track was started.")
            elif frames_processed < cfg.n_init:
                id_note = (f"Only {frames_processed} frame(s) processed; a track needs "
                           f"{cfg.n_init} consecutive matched frames (n_init) before it gets an ID.")
            else:
                id_note = ("Detections were present but no track was confirmed; detections "
                           "did not persist for n_init consecutive frames.")
            log.warning("No track IDs produced: %s", id_note)
        log.info("Done: %d frames processed (%s), %d detections in %d frames, %d IDs, %.2f FPS",
                 frames_processed, stop_reason, det_total, frames_with_dets, len(ids_seen),
                 frames_processed / elapsed if elapsed else 0.0)
        lat = np.asarray(latencies) * 1000
        return {
            "video": {"frame_count": frames_processed, "fps": round(meta["fps"], 3),
                      "duration_s": round(frames_processed / meta["fps"], 3),
                      "width": meta["width"], "height": meta["height"],
                      "reported_frame_count": reported or None,
                      "stop_reason": stop_reason, "max_frames": max_frames or None,
                      "skipped_unreadable_frames": skipped_unreadable,
                      "skipped_malformed_frames": skipped_malformed,
                      "ended_before_reported_length": bool(stop_reason == "end_of_stream" and reported
                                                           and frames_read < reported)},
            "detection": {
                "total_detections": det_total,
                "detections_above_high_conf": det_high,
                "detections_above_new_track_conf": det_new,
                "frames_with_detections": frames_with_dets,
                "max_detections_in_frame": det_max,
                "mean_detections_per_frame": round(det_total / frames_processed, 3),
            },
            "tracking": {
                "unique_track_ids": sorted(ids_seen),
                "num_unique_track_ids": len(ids_seen),
                "no_ids_reason": id_note,
                "total_tracks_created": st.tracks_confirmed,
                "final_track_states": tracker.state_counts(),
                "tracks_recovered_after_occlusion": len(st.recovered_ids),
                "recovery_events": st.recoveries,
                "recovery_events_on_confirmation": st.recoveries_on_confirmation,
                "occlusion_events": st.occlusion_events,
                "ambiguous_matches_rejected": st.ambiguous_rejections,
                "duplicate_detections_suppressed": st.duplicate_suppressions,
                "id_switches": None,
                "id_switches_note": "Not measurable without ground-truth identities; "
                                    "score /jobs/{id}/tracks against MOTChallenge GT offline.",
            },
            "output": {
                "filename": writer_info.path.name,
                "codec": writer_info.codec,
                "container": writer_info.path.suffix.lstrip("."),
                "writer_backend": writer_info.backend,
                "frames_written": frames_processed,
                "frames_decoded_on_validation": decoded,
                "size_bytes": out_size,
            },
            "performance": {
                "total_processing_time_s": round(elapsed, 3),
                "average_processing_fps": round(frames_processed / elapsed, 3) if elapsed else None,
                "latency_ms_mean": round(float(lat.mean()), 2),
                "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
                "stage_time_s": {k: round(v, 3) for k, v in timer.totals.items()},
                "peak_rss_mb": round(peak_rss / 2**20, 1) if peak_rss else None,
                "reid_backend": self.backend.info(),
                "detector": getattr(self.detector, "name", "unknown"),
            },
        }
