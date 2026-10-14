"""Backend worker entry point.

    python worker.py                      start the backend API server (same as source/main.py;
                                          host/port from PS2_HOST / PS2_PORT, default 127.0.0.1:8000)
    python worker.py path/to/video.mp4    process one video directly, without the API

Reuses the shared backend pipeline - configuration (``source/config``), the model
registry (``source/models/model_registry.py``), automatic ReID backend selection
(``source/logic/inference_manager.py``) and ``VideoProcessor``
(``source/logic/video_processor.py``) - exactly as the API's job manager does.

    python worker.py path/to/video.mp4 [--out outputs/cli]
                     [--backend auto|tensorrt|onnx-cpu|...] [--mode baseline] [--max-frames N]

Writes ``tracked.<ext>``, ``tracks_mot.txt``, ``summary.json`` and ``job.json`` (status,
progress, error) to ``--out``. Exit code 0 on success, 1 if the job failed, 2 for invalid
arguments/input. A failed job never leaves a result that looks successful.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

from source.config.settings import APP, REID  # noqa: E402
from source.logic.inference_manager import ORDER  # noqa: E402
from source.logic.job_manager import JobStatus, build_tracker_config  # noqa: E402
from source.models.model_registry import ModelRegistry  # noqa: E402
from source.utils.logging_utils import configure_logging  # noqa: E402

log = logging.getLogger("ps2.worker")

BACKEND_CHOICES = ["auto", *ORDER, "onnx", "pytorch"]


def _write_state(out_dir: Path, state: dict):
    try:
        (out_dir / "job.json").write_text(json.dumps(state, indent=2, default=str),
                                          encoding="utf-8")
    except OSError as exc:  # pragma: no cover
        log.warning("Could not persist job state: %s", exc)


def run_job(video: Path, out_dir: Path, backend: str = REID.backend, mode: str = "enhanced",
            max_frames: int | None = None, registry: ModelRegistry | None = None) -> dict:
    """Validate the input, process it with the shared pipeline and return the job state."""
    from source.logic.video_processor import VideoError, VideoProcessor, probe_video

    out_dir.mkdir(parents=True, exist_ok=True)
    state = {"input": str(video), "output_dir": str(out_dir), "backend_requested": backend,
             "mode": mode, "status": JobStatus.QUEUED.value, "frames_processed": 0,
             "total_frames": 0, "error": None, "started_at": time.time()}
    _write_state(out_dir, state)
    registry = registry or ModelRegistry()
    try:
        meta = probe_video(video)  # validate before loading any model
        state.update(status=JobStatus.PROCESSING.value, total_frames=meta["frame_count"],
                     video=meta)
        _write_state(out_dir, state)
        reid = registry.reid(backend)
        state["reid_backend"] = reid.info()
        log.info("ReID backend: %s", reid.info())
        processor = VideoProcessor(registry.detector(), reid,
                                   build_tracker_config({"tracker_mode": mode}), REID,
                                   inference_lock=registry.inference_lock)

        def progress(done: int, total: int):
            state["frames_processed"], state["total_frames"] = done, total
            if done % 50 == 0:
                _write_state(out_dir, state)

        summary = processor.process(video, out_dir / "tracked.mp4", out_dir / "tracks_mot.txt",
                                    progress_cb=progress, max_frames=max_frames)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        state.update(status=JobStatus.COMPLETED.value, summary_file="summary.json",
                     output_video=summary["output"]["filename"],
                     frames_processed=summary["video"]["frame_count"],
                     reid_backend=reid.info())
        state["summary"] = summary
    except VideoError as exc:
        state.update(status=JobStatus.FAILED.value, error=f"Invalid video: {exc}")
        log.error("Job failed: %s", state["error"])
    except Exception as exc:  # noqa: BLE001 - every failure is reported on the job
        state.update(status=JobStatus.FAILED.value, error=f"{type(exc).__name__}: {exc}")
        log.error("Job failed with exception\n%s", traceback.format_exc())
    finally:
        state["finished_at"] = time.time()
        state["processing_time_s"] = round(state["finished_at"] - state["started_at"], 3)
        if state["status"] != JobStatus.COMPLETED.value:
            for f in list(out_dir.glob("tracked.*")) + [out_dir / "tracks_mot.txt",
                                                        out_dir / "summary.json"]:
                f.unlink(missing_ok=True)  # never leave partial results behind
        _write_state(out_dir, {k: v for k, v in state.items() if k != "summary"})
        registry.release()
    return state


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Without arguments: start the backend API server. With a video: track "
                    "people in that video directly (no API).")
    ap.add_argument("video", type=Path, nargs="?",
                    help="input video to process directly; omit to start the API server")
    ap.add_argument("--out", type=Path, default=APP.output_dir / "cli")
    ap.add_argument("--backend", default=REID.backend, choices=BACKEND_CHOICES)
    ap.add_argument("--mode", default="enhanced", choices=["enhanced", "baseline"])
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)
    configure_logging()

    if args.video is None:
        from source.main import run as run_server

        log.info("No video given; starting the backend API server.")
        run_server()
        return 0
    video = args.video.resolve()
    if not video.is_file():
        print(f"error: input video not found: {args.video}", file=sys.stderr)
        return 2
    if video.suffix.lower() not in APP.allowed_extensions:
        print(f"error: unsupported file type '{video.suffix}' (allowed: "
              f"{', '.join(APP.allowed_extensions)})", file=sys.stderr)
        return 2
    if args.max_frames is not None and args.max_frames <= 0:
        print("error: --max-frames must be positive", file=sys.stderr)
        return 2

    state = run_job(video, args.out.resolve(), args.backend, args.mode, args.max_frames)
    if state["status"] != JobStatus.COMPLETED.value:
        print(f"FAILED: {state['error']}", file=sys.stderr)
        return 1
    summary = state["summary"]
    print(json.dumps({k: summary[k] for k in ("video", "detection", "output", "performance")},
                     indent=2))
    print(f"output video: {args.out / summary['output']['filename']}")
    t = summary["tracking"]
    if t["no_ids_reason"]:
        print(f"no IDs: {t['no_ids_reason']}")
    print(f"unique IDs: {t['num_unique_track_ids']}  recovered: {t['tracks_recovered_after_occlusion']}"
          f"  occlusion events: {t['occlusion_events']}  ambiguous rejected: "
          f"{t['ambiguous_matches_rejected']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
