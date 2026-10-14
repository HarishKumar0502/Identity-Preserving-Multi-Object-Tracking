"""Video writer selection and output validation.

The writer tries an ordered list of ``FOURCC[@API]:.container`` candidates (default
``H264@MSMF:.mp4,mp4v:.mp4,MJPG:.avi``). Each candidate must pass a *preflight*: open,
write a few frames to a scratch file, reopen and decode them, before it is used,
so an encoder that "opens" but produces unreadable files is skipped.

* ``H264@MSMF`` - H.264 through Windows Media Foundation (part of Windows, no extra
  DLL). Browser-playable MP4. Unavailable on non-Windows / "N" editions: skipped.
* ``mp4v``      - MPEG-4 Part 2 via OpenCV's bundled FFmpeg. Plays in VLC, not in
  most browsers.
* ``MJPG/.avi`` - last resort.

H.264 through OpenCV's FFmpeg backend (``avc1`` without ``@MSMF``) is *not* used: the
opencv-python wheels delegate it to Cisco's ``openh264-*.dll``, which is not shipped,
which is exactly what printed "Failed to load OpenH264 library: openh264-2.5.0-win64.dll".
No third-party binary is downloaded. Override with ``PS2_VIDEO_CODECS``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2

from source.config.settings import APP

log = logging.getLogger("ps2.video")

MEDIA_TYPES = {".mp4": "video/mp4", ".avi": "video/x-msvideo"}


class VideoWriterError(RuntimeError):
    pass


@dataclass(frozen=True)
class WriterInfo:
    codec: str
    path: Path
    fps: float
    width: int
    height: int
    backend: str


_APIS = {"MSMF": "CAP_MSMF", "FFMPEG": "CAP_FFMPEG", "ANY": "CAP_ANY"}
PREFLIGHT_FRAMES = 3


def parse_codec_candidates(spec: str) -> Tuple[Tuple[str, ...], ...]:
    """'H264@MSMF:.mp4,mp4v:.mp4' -> (('H264', '.mp4', 'MSMF'), ('mp4v', '.mp4'))."""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        codec, _, ext = item.partition(":")
        fourcc, _, api = codec.strip().partition("@")
        ext, api = ext.strip().lower(), api.strip().upper()
        if len(fourcc) != 4 or not ext.startswith(".") or (api and api not in _APIS):
            raise ValueError(f"Invalid codec candidate '{item}' (expected FOURCC[@API]:.ext, "
                             f"API one of {sorted(_APIS)}).")
        out.append((fourcc, ext, api) if api else (fourcc, ext))
    if not out:
        raise ValueError("No codec candidates configured.")
    return tuple(out)


def _open_writer(path: Path, fourcc: str, api: str, fps: float, size: Tuple[int, int]):
    code = cv2.VideoWriter_fourcc(*fourcc)
    if api:
        api_id = getattr(cv2, _APIS[api], None)
        if api_id is None:
            return None
        return cv2.VideoWriter(str(path), api_id, code, float(fps), size)
    return cv2.VideoWriter(str(path), code, float(fps), size)


def _preflight(path: Path, fourcc: str, api: str, fps: float, size: Tuple[int, int]) -> str:
    """Write and decode a few frames with this encoder; return '' if OK else the reason."""
    import numpy as np

    probe = path.with_name(f".preflight_{path.stem}_{fourcc}{path.suffix}")
    writer = None
    try:
        writer = _open_writer(probe, fourcc, api, fps, size)
        if writer is None or not writer.isOpened():
            return "encoder could not be opened"
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        for i in range(PREFLIGHT_FRAMES):
            frame[:] = 40 * (i + 1)
            writer.write(frame)
        writer.release()
        writer = None
        validate_video_output(probe, expected_frames=PREFLIGHT_FRAMES, size=size)
        return ""
    except (VideoWriterError, cv2.error) as exc:
        return f"preflight failed: {exc}"
    finally:
        if writer is not None:
            writer.release()
        try:
            probe.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - file briefly locked by a scanner
            pass


def create_video_writer(output_path: Path, fps: float, size: Tuple[int, int],
                        candidates: Optional[Sequence[Tuple[str, ...]]] = None,
                        preflight: bool = True) -> Tuple[cv2.VideoWriter, WriterInfo]:
    """Open the first working (fourcc, container) candidate.

    ``output_path``'s suffix is replaced by each candidate's container, so the
    returned ``WriterInfo.path`` is the file actually being written.
    """
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise VideoWriterError(f"Invalid output frame size {width}x{height}.")
    if not fps or fps <= 0:
        raise VideoWriterError(f"Invalid output FPS {fps}.")
    candidates = list(candidates or parse_codec_candidates(APP.video_codecs))
    tried = []
    for cand in candidates:
        fourcc, ext, api = (tuple(cand) + ("",))[:3]
        label = f"{fourcc}@{api}" if api else fourcc
        path = Path(output_path).with_suffix(ext)
        reason = _preflight(path, fourcc, api, fps, (width, height)) if preflight else ""
        writer = None
        if not reason:
            writer = _open_writer(path, fourcc, api, fps, (width, height))
            if writer is not None and writer.isOpened():
                try:
                    backend = writer.getBackendName()
                except cv2.error:  # pragma: no cover - backend name is informational
                    backend = api or "unknown"
                info = WriterInfo(fourcc, path, float(fps), width, height, backend)
                log.info("VideoWriter opened: codec=%s backend=%s fps=%.3f size=%dx%d path=%s",
                         fourcc, backend, fps, width, height, path)
                return writer, info
            reason = "encoder could not be opened"
        if writer is not None:
            writer.release()
        if path.is_file() and path.stat().st_size == 0:
            path.unlink(missing_ok=True)  # don't leave an empty stub behind
        tried.append(f"{label}/{ext}: {reason}")
        log.warning("VideoWriter candidate %s (%s) unusable: %s; trying next", label, ext, reason)
    raise VideoWriterError("No supported video encoder could be initialized "
                           f"(tried {'; '.join(tried)}).")


def validate_video_output(path: Path, expected_frames: Optional[int] = None,
                          size: Optional[Tuple[int, int]] = None) -> int:
    """Reopen a written video and decode it; return the decoded frame count.

    Raises ``VideoWriterError`` if the file is missing/empty, cannot be opened,
    has no decodable frame, has the wrong frame size, or (when given) decodes
    fewer frames than were written.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise VideoWriterError("Output video was not written (missing or empty file).")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoWriterError("Output video could not be reopened for validation.")
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if n == 0 and size is not None and (frame.shape[1], frame.shape[0]) != tuple(size):
                raise VideoWriterError(f"Output frame size {frame.shape[1]}x{frame.shape[0]} "
                                       f"does not match expected {size[0]}x{size[1]}.")
            n += 1
    finally:
        cap.release()
    if n == 0:
        raise VideoWriterError("Output video contains no decodable frames.")
    if expected_frames is not None and n < expected_frames:
        raise VideoWriterError(f"Output video is incomplete: decoded {n} of "
                               f"{expected_frames} written frames.")
    return n
