"""Centralized configuration for the PS-2 occlusion-resistant MOT backend.

Every tunable used by the detector, ReID backends, tracker, worker and API lives
here. Values can be overridden with environment variables prefixed ``PS2_`` (see
``_env``), or with a ``.env`` file at the project root (see ``.env.example``; real
environment variables win). They are internal defaults: the API only takes a video file.

All paths are absolute (anchored at the project root), so the API and the worker work
from any current directory.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "source"
MODELS_DIR = SOURCE_DIR / "models"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Minimal ``KEY=VALUE`` loader (no extra dependency); never overrides real env vars."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


load_dotenv()


def _env(name: str, default):
    raw = os.environ.get(f"PS2_{name}")
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def _default_reid_checkpoint() -> Path:
    """Locate the PS-1 checkpoint: env override, then source/models/reid/."""
    override = os.environ.get("PS2_REID_CHECKPOINT")
    if override:
        return Path(override)
    for candidate in (
        MODELS_DIR / "reid" / "best.pth",
        MODELS_DIR / "reid" / "ps1_resnet18_3epoch_best.pth",
        MODELS_DIR / "reid" / "ps1_resnet18_baseline.pth",
    ):
        if candidate.exists():
            return candidate
    return MODELS_DIR / "reid" / "best.pth"


# "auto" selects the best verified backend at start-up (source/logic/inference_manager.py).
# Explicit names force one candidate: tensorrt | onnx-cuda | onnx-dml | pytorch-cuda |
# onnx-cpu | pytorch-cpu, or the legacy "onnx" / "pytorch" (best device for that runtime).
ReIDBackendName = str


@dataclass(frozen=True)
class ReIDConfig:
    # PS-1 checkpoint: torchvision ResNet-18 state dict under "model_state_dict",
    # with a 751-way CosFace head stored as `fc` (unit-norm rows, zero bias).
    checkpoint: Path = field(default_factory=_default_reid_checkpoint)
    architecture: str = "resnet18"
    num_train_classes: int = 751
    # Input size (H, W). Standard Market-1501 ReID resolution. The PS-1 training
    # source was not provided, so this is the documented assumption; change via
    # PS2_REID_INPUT_H / PS2_REID_INPUT_W if PS-1 used a different size.
    input_size: Tuple[int, int] = (_env("REID_INPUT_H", 256), _env("REID_INPUT_W", 128))
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet (RGB)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    embedding_dim: int = 512
    metric: Literal["cosine"] = "cosine"
    backend: ReIDBackendName = _env("REID_BACKEND", "auto")
    device: str = _env("DEVICE", "auto")  # auto | cpu | cuda | cuda:0
    batch_size: int = _env("REID_BATCH", 32)
    onnx_path: Path = MODELS_DIR / "onnx" / "reid_resnet18.onnx"
    tensorrt_dir: Path = Path(_env("TENSORRT_DIR", str(MODELS_DIR / "tensorrt")))
    tensorrt_precision: str = _env("TENSORRT_PRECISION", "fp32")  # fp32 | fp16 (validated)
    tensorrt_max_batch: int = 32
    tensorrt_opt_batch: int = 8
    # Build the engine at start-up when it is missing or stale (one-off, then cached).
    tensorrt_auto_build: bool = _env("TENSORRT_AUTO_BUILD", True)
    # A candidate backend is accepted only if its embeddings for a fixed probe batch
    # agree with the PyTorch reference to at least this cosine similarity (see README).
    verify_min_cosine: float = _env("REID_VERIFY_MIN_COSINE", 0.9995)
    # Crops smaller than this are rejected before inference.
    min_crop_h: int = 24
    min_crop_w: int = 10

    @property
    def tensorrt_path(self) -> Path:
        return self.tensorrt_dir / f"reid_resnet18_{self.tensorrt_precision}.engine"


@dataclass(frozen=True)
class DetectorConfig:
    weights: Path = Path(_env("DETECTOR_WEIGHTS", str(MODELS_DIR / "detector" / "yolo11n.pt")))
    person_class_id: int = 0  # COCO "person"
    # Detector runs at the low threshold so the tracker's second (low-score)
    # association stage can use weak, typically occluded, boxes.
    min_confidence: float = 0.10
    iou_nms: float = 0.6
    imgsz: int = _env("DETECTOR_IMGSZ", 640)
    device: str = _env("DEVICE", "auto")
    half: bool = False


@dataclass(frozen=True)
class TrackerConfig:
    """Association / lifecycle parameters. See source/logic/tracking/track_manager.py."""

    # Detection score split (ByteTrack-style two-stage association).
    high_conf: float = 0.50
    new_track_conf: float = 0.60
    # Lifecycle (frames).
    n_init: int = 3                 # consecutive hits to confirm a track
    max_occluded_frames: int = 15   # missed this long -> occluded, then lost
    track_buffer: int = 90          # lost tracks kept this long for re-association
    max_lost_tracks: int = 200      # hard cap on lost-track memory (oldest dropped)
    # Gating.
    use_appearance: bool = True
    chi2_gate: float = 9.4877       # 95% chi-square, 4 DoF (x, y, a, h)
    appearance_gate: float = 0.45   # max cosine distance for an active match ...
    iou_trust: float = 0.50         # ... unless IoU >= this (strong spatial evidence),
    appearance_veto: float = 0.90   # in which case only distances above this veto
    min_iou_without_appearance: float = 0.10
    # Cost weights (each component normalized to [0, 1]).
    w_motion: float = 0.20
    w_appearance: float = 0.50
    w_iou: float = 0.30
    match_thresh: float = 0.60      # max fused cost for an accepted active match
    low_conf_iou_thresh: float = 0.50   # stage-2 (low score) needs 1-IoU <= this
    tentative_iou_thresh: float = 0.70  # stage-3 needs 1-IoU <= this
    # Re-association of lost tracks.
    enable_reassociation: bool = True
    reid_thresh: float = 0.25       # max cosine distance to restore a lost ID (see README calibration)
    ambiguity_ratio: float = 0.85   # best must beat 2nd-best by this ratio
    reassoc_radius: float = 1.5     # search radius, in track heights, at loss
    reassoc_radius_growth: float = 0.08  # extra heights per lost frame
    reassoc_radius_max: float = 4.0
    # Appearance memory.
    gallery_size: int = 30
    ema_alpha: float = 0.9
    feature_min_conf: float = 0.60
    feature_max_overlap: float = 0.30   # skip update if box is >30% covered by others
    feature_border_margin: int = 3      # boxes this close to the frame edge are truncated crops
    feature_outlier_dist: float = 0.50  # skip update if far from the track's identity
    ambiguous_margin: float = 0.05      # skip update if 2nd-best match is this close
    duplicate_iou: float = 0.80         # suppress new tracks overlapping a live track
    # Rendering.
    draw_predicted_occluded: bool = True

    def with_overrides(self, **kwargs) -> "TrackerConfig":
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)


def baseline_tracker_config() -> TrackerConfig:
    """SORT-like baseline: IoU + motion only, short memory, no ReID recovery."""
    return TrackerConfig(
        use_appearance=False,
        enable_reassociation=False,
        w_motion=0.0,
        w_appearance=0.0,
        w_iou=1.0,
        match_thresh=0.70,
        low_conf_iou_thresh=0.0,  # disables the low-score stage
        max_occluded_frames=3,
        track_buffer=0,
    )


@dataclass(frozen=True)
class AppConfig:
    upload_dir: Path = UPLOAD_DIR
    output_dir: Path = OUTPUT_DIR
    allowed_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")
    max_upload_mb: int = _env("MAX_UPLOAD_MB", 500)
    max_concurrent_jobs: int = _env("MAX_CONCURRENT_JOBS", 1)
    max_queued_jobs: int = _env("MAX_QUEUED_JOBS", 8)
    job_ttl_hours: float = _env("JOB_TTL_HOURS", 24.0)
    max_frames: Optional[int] = None  # optional hard cap per video (None = whole video)
    # Output encoder candidates, tried in order (FOURCC[@API]:.container); see
    # source/utils/video_utils.py. H264@MSMF = Windows Media Foundation H.264 (browser-playable,
    # part of Windows, no OpenH264 DLL); others are fallbacks.
    video_codecs: str = _env("VIDEO_CODECS", "H264@MSMF:.mp4,mp4v:.mp4,MJPG:.avi")
    # Multipart uploads are spooled here (not the system %TEMP%, which may be small).
    temp_dir: Path = Path(_env("TEMP_DIR", str(UPLOAD_DIR / ".tmp")))
    # The uploaded input is deleted once its job has finished (outputs are kept until TTL).
    keep_inputs: bool = _env("KEEP_INPUTS", False)


REID = ReIDConfig()
DETECTOR = DetectorConfig()
TRACKER = TrackerConfig()
APP = AppConfig()


def as_dict(cfg) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
