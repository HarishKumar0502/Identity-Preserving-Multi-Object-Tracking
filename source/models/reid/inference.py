"""PS-1 ReID model: checkpoint loading, shared preprocessing and the PyTorch backend.

Every ReID backend (PyTorch here, ONNX Runtime in ``models/onnx``, TensorRT in
``models/tensorrt``) derives from ``ReIDBackend``, shares ``preprocess_crops`` and returns
L2-normalized (N, 512) float32 embeddings, so the tracker never depends on which backend
is active.

PS-1 checkpoint layout (verified by inspecting ``best.pth``)::

    {"epoch": int, "val_tar_at_far": float, "experiment": {...},
     "model_state_dict": <torchvision resnet18 state dict, fc: Linear(512, 751)>}

The ``fc`` layer is the CosFace training head (unit-norm rows, zero bias). For
feature extraction it is replaced by ``Identity`` *after* a strict load, so the
embedding is the 512-d global-average-pooled backbone output.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from source.config.settings import REID, ReIDConfig
from source.utils.device_utils import resolve_device

log = logging.getLogger("ps2.reid")


class ReIDModelError(RuntimeError):
    """Checkpoint missing, incompatible, or backend unavailable."""


# --------------------------------------------------------------------------- #
# Preprocessing (shared by every backend)
# --------------------------------------------------------------------------- #
def preprocess_crops(crops: Sequence[np.ndarray], cfg: ReIDConfig = REID) -> np.ndarray:
    """BGR uint8 crops -> (N, 3, H, W) float32, RGB, ImageNet-normalized."""
    h, w = cfg.input_size
    out = np.empty((len(crops), 3, h, w), dtype=np.float32)
    mean = np.asarray(cfg.mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(cfg.std, dtype=np.float32).reshape(3, 1, 1)
    for i, crop in enumerate(crops):
        resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out[i] = (rgb.transpose(2, 0, 1) - mean) / std
    return out


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


# --------------------------------------------------------------------------- #
# PyTorch model reconstruction
# --------------------------------------------------------------------------- #
def load_checkpoint_state(path: Path) -> dict:
    import torch

    path = Path(path)
    if not path.is_file():
        raise ReIDModelError(f"ReID checkpoint not found: {path.name}. Place the PS-1 checkpoint "
                             f"at source/models/reid/best.pth or set PS2_REID_CHECKPOINT.")
    # weights_only=True first; the PS-1 file contains only tensors and plain types.
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict) or not ckpt:
        raise ReIDModelError("Checkpoint does not contain a state dict.")
    # Strip DataParallel / wrapper prefixes if present.
    for prefix in ("module.", "backbone.", "model."):
        if all(k.startswith(prefix) for k in ckpt):
            ckpt = {k[len(prefix):]: v for k, v in ckpt.items()}
    return ckpt


def build_reid_model(cfg: ReIDConfig = REID):
    """Rebuild the PS-1 ResNet-18, strict-load weights, then drop the classifier."""
    import torch
    import torchvision

    if cfg.architecture != "resnet18":
        raise ReIDModelError(f"Unsupported ReID architecture '{cfg.architecture}'.")
    state = load_checkpoint_state(cfg.checkpoint)
    num_classes = state["fc.weight"].shape[0] if "fc.weight" in state else cfg.num_train_classes
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ReIDModelError(f"Checkpoint is incompatible with torchvision resnet18: {exc}") from exc
    if model.fc.in_features != cfg.embedding_dim:
        raise ReIDModelError(f"Expected {cfg.embedding_dim}-d features, got {model.fc.in_features}.")
    model.fc = torch.nn.Identity()  # embedding = pooled 512-d backbone feature
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class ReIDBackend(ABC):
    name: str = "base"
    device: str = "cpu"
    precision: str = "fp32"
    model_path: Optional[Path] = None

    def __init__(self, cfg: ReIDConfig = REID):
        self.cfg = cfg

    @abstractmethod
    def _infer(self, batch: np.ndarray) -> np.ndarray:
        """(B, 3, H, W) float32 -> (B, D) raw features."""

    def embed_preprocessed(self, batch: np.ndarray) -> np.ndarray:
        feats = [self._infer(np.ascontiguousarray(batch[i:i + self.cfg.batch_size]))
                 for i in range(0, len(batch), self.cfg.batch_size)]
        return l2_normalize(np.concatenate(feats, axis=0))

    def extract_embeddings(self, person_crops: Sequence[np.ndarray]) -> np.ndarray:
        """Unified entry point: BGR crops -> (N, D) L2-normalized embeddings."""
        if len(person_crops) == 0:
            return np.zeros((0, self.cfg.embedding_dim), dtype=np.float32)
        return self.embed_preprocessed(preprocess_crops(person_crops, self.cfg))

    def close(self):
        """Release device resources (sessions, engines, buffers)."""

    def info(self) -> dict:
        return {"backend": self.name, "device": self.device, "precision": self.precision,
                "model": self.model_path.name if self.model_path else None}


class PyTorchBackend(ReIDBackend):
    name = "pytorch"

    def __init__(self, cfg: ReIDConfig = REID, device: Optional[str] = None):
        super().__init__(cfg)
        import torch

        self.torch = torch
        try:
            self.device = resolve_device(device or cfg.device)
        except RuntimeError as exc:
            raise ReIDModelError(str(exc)) from exc
        self.name = f"pytorch-{'cuda' if self.device.startswith('cuda') else 'cpu'}"
        self.model_path = Path(cfg.checkpoint)
        self.model = build_reid_model(cfg).to(self.device)

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(batch).to(self.device, non_blocking=True)
            return self.model(x).float().cpu().numpy()

    def close(self):
        self.model = None
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
