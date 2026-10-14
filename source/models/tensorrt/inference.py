"""TensorRT backend for the PS-1 ReID model (engines in this folder).

Engines are built from ``models/onnx/reid_resnet18.onnx`` by ``engine_builder.py``
(auto-built at start-up when missing or stale, or with ``scripts/build_reid_tensorrt.py``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from source.config.settings import REID, ReIDConfig
from source.models.reid.inference import ReIDBackend, ReIDModelError


class TensorRTBackend(ReIDBackend):
    """Native TensorRT engine runner (TensorRT >= 8.6 tensor API).

    Device buffers come from the CUDA driver API (``cuda-python``), so no CUDA torch
    build is needed. The engine is built/validated by ``engine_builder.py``
    (auto-built at start-up when missing or stale, if ``cfg.tensorrt_auto_build``).
    """

    name = "tensorrt"

    def __init__(self, cfg: ReIDConfig = REID, path: Optional[Path] = None,
                 auto_build: Optional[bool] = None):
        super().__init__(cfg)
        from source.utils.device_utils import (CudaDevice, CudaError,
                                                ensure_gpu_runtime_on_path, probe_cuda)

        self._dev = None
        self._in = self._out = self._stream = 0
        self.context = self.engine = None
        ensure_gpu_runtime_on_path()
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ReIDModelError("TensorRT Python package is not installed.") from exc
        gpu = probe_cuda()
        if not gpu["available"]:
            raise ReIDModelError(f"TensorRT needs a CUDA GPU: {gpu['reason']}")
        self.trt = trt
        self.precision = cfg.tensorrt_precision
        self.build_meta = None
        if path is None:
            from source.models.tensorrt.engine_builder import EngineBuildError, ensure_engine
            auto = cfg.tensorrt_auto_build if auto_build is None else auto_build
            try:
                path, self.build_meta = ensure_engine(
                    cfg.onnx_path, cfg.tensorrt_path, cfg.tensorrt_precision, cfg.input_size,
                    cfg.tensorrt_max_batch, cfg.tensorrt_opt_batch, auto)
            except EngineBuildError as exc:
                raise ReIDModelError(str(exc)) from exc
        path = Path(path)
        if not path.is_file():
            raise ReIDModelError(f"TensorRT engine not found ({path.name}); run "
                                 "scripts/build_reid_tensorrt.py on this machine.")
        self.model_path = path
        self.device = f"cuda ({gpu['name']})"
        try:
            self._dev = CudaDevice()
            self._dev.activate()
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
            self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
            if self.engine is None:
                raise ReIDModelError("Failed to deserialize the TensorRT engine (corrupted, or built "
                                     "for a different GPU / TensorRT version). Rebuild it with "
                                     "scripts/build_reid_tensorrt.py.")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise ReIDModelError("Could not create a TensorRT execution context "
                                     "(GPU out of memory?).")
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            mode = trt.TensorIOMode
            self.input_name = next(n for n in names if self.engine.get_tensor_mode(n) == mode.INPUT)
            self.output_name = next(n for n in names if self.engine.get_tensor_mode(n) == mode.OUTPUT)
            shape = tuple(self.engine.get_tensor_shape(self.input_name))
            if tuple(shape[1:]) != (3, *cfg.input_size):
                raise ReIDModelError(f"Engine input shape {shape} does not match the configured "
                                     f"ReID input (3, {cfg.input_size[0]}, {cfg.input_size[1]}).")
            self.max_batch = int(self.engine.get_tensor_profile_shape(self.input_name, 0)[2][0])
            h, w = cfg.input_size
            self._in = self._dev.malloc(self.max_batch * 3 * h * w * 4)
            self._out = self._dev.malloc(self.max_batch * cfg.embedding_dim * 4)
            self._stream = self._dev.stream()
            self.context.set_tensor_address(self.input_name, self._in)
            self.context.set_tensor_address(self.output_name, self._out)
        except CudaError as exc:
            self.close()
            raise ReIDModelError(f"CUDA error while loading the TensorRT engine: {exc}") from exc
        except BaseException:
            self.close()
            raise

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        from source.utils.device_utils import CudaError

        outs = []
        try:
            self._dev.activate()  # the worker thread differs from the loading thread
            for i in range(0, len(batch), self.max_batch):
                chunk = np.ascontiguousarray(batch[i:i + self.max_batch], dtype=np.float32)
                out = np.empty((len(chunk), self.cfg.embedding_dim), dtype=np.float32)
                self.context.set_input_shape(self.input_name, tuple(chunk.shape))
                self._dev.htod(self._in, chunk, self._stream)
                if not self.context.execute_async_v3(self._stream):
                    raise ReIDModelError("TensorRT execution failed.")
                self._dev.dtoh(out, self._out, self._stream)
                self._dev.sync(self._stream)
                outs.append(out)
        except CudaError as exc:
            raise ReIDModelError(f"CUDA error during TensorRT inference: {exc}") from exc
        return np.concatenate(outs, axis=0)

    def close(self):
        self.context = self.engine = None
        if self._dev is not None:
            for ptr in (self._in, self._out):
                try:
                    self._dev.free(ptr)
                except Exception:  # noqa: BLE001 - best effort during teardown
                    pass
            try:
                self._dev.destroy_stream(self._stream)
            except Exception:  # noqa: BLE001
                pass
        self._in = self._out = self._stream = 0

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
