"""ONNX Runtime backend for the PS-1 ReID model (``reid_resnet18.onnx`` in this folder).

The ONNX graph is exported from ``models/reid/best.pth`` by ``scripts/export_reid_onnx.py``
(opset 17, input ``input [batch,3,256,128]``, output ``embedding [batch,512]``). It is also
the source graph for the TensorRT engines in ``models/tensorrt``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from source.config.settings import REID, ReIDConfig
from source.models.reid.inference import ReIDBackend, ReIDModelError


ORT_PROVIDERS = {"cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider",
                 "cpu": "CPUExecutionProvider"}


class OnnxBackend(ReIDBackend):
    """ONNX Runtime session on one explicit execution provider (cuda | dml | cpu).

    ``provider="auto"`` (legacy) uses CUDA when available, else CPU. A requested
    accelerator that the session does not actually activate raises instead of
    silently running on CPU.
    """

    name = "onnx"

    def __init__(self, cfg: ReIDConfig = REID, path: Optional[Path] = None, provider: str = "auto"):
        super().__init__(cfg)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ReIDModelError("onnxruntime is not installed.") from exc
        path = Path(path or cfg.onnx_path)
        if not path.is_file():
            raise ReIDModelError(f"ONNX model not found ({path.name}); run scripts/export_reid_onnx.py.")
        available = ort.get_available_providers()
        if provider == "auto":
            provider = "cuda" if ORT_PROVIDERS["cuda"] in available else "cpu"
        if provider not in ORT_PROVIDERS:
            raise ReIDModelError(f"Unknown ONNX Runtime provider '{provider}'.")
        wanted = ORT_PROVIDERS[provider]
        if wanted not in available:
            raise ReIDModelError(f"ONNX Runtime {ort.__version__} has no {wanted} "
                                 f"(available: {', '.join(available)}).")
        providers = [wanted] + (["CPUExecutionProvider"] if provider != "cpu" else [])
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        try:
            self.session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        except Exception as exc:  # noqa: BLE001 - provider DLL / CUDA init failures
            raise ReIDModelError(f"ONNX Runtime could not create a {wanted} session: {exc}") from exc
        active = self.session.get_providers()
        if active[0] != wanted:
            raise ReIDModelError(f"ONNX Runtime fell back to {active[0]} instead of {wanted}.")
        self.device = {"cuda": "cuda", "dml": "directml", "cpu": "cpu"}[provider]
        self.name = f"onnx-{provider}"
        self.model_path = path
        self.input_name = self.session.get_inputs()[0].name

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: batch.astype(np.float32)})[0]

    def close(self):
        self.session = None
