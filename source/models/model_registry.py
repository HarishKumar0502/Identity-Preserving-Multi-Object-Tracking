"""Model registry: named ReID backend factories and the process-wide model cache.

* ``backend_factories`` / ``create_backend`` instantiate one named ReID backend
  (PyTorch ``models/reid``, ONNX Runtime ``models/onnx``, TensorRT ``models/tensorrt``).
* ``ModelRegistry`` lazily loads the detector (``models/detector``) and the ReID backend
  (auto-selected and verified by ``source/logic/inference_manager.py``) once per process,
  under a lock, and shares them between jobs. GPU-bound inference is serialized by
  ``inference_lock``.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, List, Optional

from source.config.settings import DETECTOR, REID, ReIDConfig
from source.models.onnx.inference import OnnxBackend
from source.models.reid.inference import PyTorchBackend, ReIDBackend, ReIDModelError
from source.models.tensorrt.inference import TensorRTBackend

log = logging.getLogger("ps2.models")


def backend_factories() -> Dict[str, Callable[[ReIDConfig], ReIDBackend]]:
    return {
        "tensorrt": lambda cfg: TensorRTBackend(cfg),
        "onnx-cuda": lambda cfg: OnnxBackend(cfg, provider="cuda"),
        "onnx-dml": lambda cfg: OnnxBackend(cfg, provider="dml"),
        "pytorch-cuda": lambda cfg: PyTorchBackend(cfg, device="cuda"),
        "onnx-cpu": lambda cfg: OnnxBackend(cfg, provider="cpu"),
        "pytorch-cpu": lambda cfg: PyTorchBackend(cfg, device="cpu"),
        # legacy names: best device for that runtime
        "onnx": lambda cfg: OnnxBackend(cfg, provider="auto"),
        "pytorch": lambda cfg: PyTorchBackend(cfg),
    }


def create_backend(name: Optional[str] = None, cfg: ReIDConfig = REID) -> ReIDBackend:
    """Instantiate one named backend (no fallback). ``"auto"`` runs the verified selection."""
    name = (name or cfg.backend).lower()
    if name == "auto":
        from source.logic.inference_manager import select_backend
        return select_backend(cfg).backend
    factories = backend_factories()
    if name not in factories:
        raise ReIDModelError(f"Unknown ReID backend '{name}'. Choose from "
                             f"{sorted(factories) + ['auto']}.")
    backend = factories[name](cfg)
    log.info("ReID backend ready: %s", backend.info())
    return backend


def available_backends() -> List[str]:
    """Candidates whose runtime dependencies are present (not yet load-verified)."""
    from source.logic.inference_manager import default_candidates

    return [c.name for c in default_candidates(REID) if c.available()[0]]


class ModelRegistry:
    """Process-wide, lazily-loaded detector and ReID backend (selected automatically)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self._detector = None
        self._reid: Dict[str, object] = {}
        self.selection: Optional[dict] = None
        self.errors: Dict[str, str] = {}

    def detector(self):
        with self._lock:
            if self._detector is None:
                from source.models.detector.inference import create_detector
                try:
                    self._detector = create_detector(DETECTOR)
                    self.errors.pop("detector", None)
                except Exception as exc:
                    self.errors["detector"] = str(exc)
                    raise
            return self._detector

    def reid(self, backend: Optional[str] = None):
        """The ReID backend: auto-selected and verified once, then cached.

        ``backend`` forces a candidate (internal use: CLI, benchmarks); the API never
        passes it. ``PS2_REID_BACKEND`` does the same process-wide.
        """
        name = (backend or REID.backend).lower()
        with self._lock:
            if name not in self._reid:
                from source.logic.inference_manager import select_backend
                try:
                    result = select_backend(REID, requested=name)
                    self._reid[name] = result.backend
                    if name == REID.backend.lower():
                        self.selection = {"requested": name, **result.report()}
                    self.errors.pop("reid", None)
                except Exception as exc:
                    self.errors["reid"] = str(exc)
                    raise
            return self._reid[name]

    def status(self) -> dict:
        active = self._reid.get(REID.backend.lower())
        return {"detector_loaded": self._detector is not None,
                "reid_loaded": active is not None,
                "reid_backend": active.info() if active is not None else None,
                "reid_selection": self.selection,
                "errors": dict(self.errors)}

    def release(self):
        with self._lock:
            self._detector = None
            for backend in self._reid.values():
                try:
                    backend.close()
                except Exception:  # noqa: BLE001 - best effort
                    pass
            self._reid.clear()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
