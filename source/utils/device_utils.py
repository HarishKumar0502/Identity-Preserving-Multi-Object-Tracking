"""Device selection and optional GPU runtime discovery (TensorRT, CUDA driver).

``resolve_device`` maps the configured torch device (``auto`` | ``cpu`` | ``cuda[:n]``).

TensorRT 11 + ``cuda-python`` may be installed with ``pip --target`` into a
project-local directory (default ``<project>/.gpu_runtime``, override with
``PS2_GPU_RUNTIME``) when the system drive has no room for the ~3.3 GB TensorRT
libraries. ``ensure_gpu_runtime_on_path()`` appends that directory to ``sys.path``
(after site-packages, so it never shadows existing packages) only if it exists.

Device memory for the TensorRT runner comes from the CUDA *driver* API
(``cuda.bindings.driver`` -> ``nvcuda.dll``, shipped with the NVIDIA display driver),
so a CPU-only torch install is fine.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

from source.config.settings import PROJECT_ROOT

log = logging.getLogger("ps2.gpu")


def resolve_device(requested: str) -> str:
    """Map 'auto' to cuda when torch has a usable CUDA device, else cpu."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Device '{requested}' requested but torch has no CUDA support "
                           f"(torch {torch.__version__}).")
    return requested


def gpu_runtime_dir() -> Path:
    return Path(os.environ.get("PS2_GPU_RUNTIME", PROJECT_ROOT / ".gpu_runtime"))


def ensure_gpu_runtime_on_path() -> Optional[Path]:
    d = gpu_runtime_dir()
    if d.is_dir() and str(d) not in sys.path:
        sys.path.append(str(d))
    return d if d.is_dir() else None


class CudaError(RuntimeError):
    pass


def _driver():
    ensure_gpu_runtime_on_path()
    try:
        from cuda.bindings import driver as cu
    except ImportError as exc:
        raise CudaError("cuda-python (cuda.bindings) is not installed.") from exc
    return cu


def check(result):
    """Unpack a cuda.bindings call result, raising CudaError on failure."""
    cu = _driver()
    err, *rest = result if isinstance(result, tuple) else (result,)
    if err != cu.CUresult.CUDA_SUCCESS:
        _, msg = cu.cuGetErrorString(err)
        raise CudaError(f"{err.name}: {msg.decode() if isinstance(msg, bytes) else msg}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else tuple(rest)


@lru_cache(maxsize=1)
def probe_cuda() -> dict:
    """{'available': bool, 'reason'|..., 'name', 'compute_capability', 'total_mem_mb', 'driver_cuda'}."""
    try:
        cu = _driver()
        check(cu.cuInit(0))
        if check(cu.cuDeviceGetCount()) < 1:
            return {"available": False, "reason": "No CUDA device found."}
        dev = check(cu.cuDeviceGet(0))
        name = check(cu.cuDeviceGetName(128, dev)).split(b"\0")[0].decode()
        attr = cu.CUdevice_attribute
        major = check(cu.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev))
        minor = check(cu.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev))
        ver = check(cu.cuDriverGetVersion())
        return {"available": True, "name": name, "compute_capability": f"{major}.{minor}",
                "total_mem_mb": int(check(cu.cuDeviceTotalMem(dev)) // 2**20),
                "driver_cuda": f"{ver // 1000}.{(ver % 1000) // 10}"}
    except CudaError as exc:
        return {"available": False, "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 - any loader failure means "no CUDA"
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def tensorrt_version() -> Optional[str]:
    ensure_gpu_runtime_on_path()
    try:
        import tensorrt as trt
        return trt.__version__
    except Exception:  # noqa: BLE001 - ImportError or a broken DLL load
        return None


class CudaDevice:
    """Primary context of device 0, made current in whichever thread calls ``activate``."""

    _lock = threading.Lock()
    _ctx = None

    def __init__(self):
        cu = _driver()
        with CudaDevice._lock:
            if CudaDevice._ctx is None:
                check(cu.cuInit(0))
                dev = check(cu.cuDeviceGet(0))
                CudaDevice._ctx = check(cu.cuDevicePrimaryCtxRetain(dev))
        self.cu = cu
        self.ctx = CudaDevice._ctx

    def activate(self):
        check(self.cu.cuCtxSetCurrent(self.ctx))

    def malloc(self, nbytes: int) -> int:
        self.activate()
        return int(check(self.cu.cuMemAlloc(nbytes)))

    def free(self, ptr: int):
        if ptr:
            self.activate()
            check(self.cu.cuMemFree(ptr))

    def stream(self) -> int:
        self.activate()
        return int(check(self.cu.cuStreamCreate(0)))

    def destroy_stream(self, stream: int):
        if stream:
            self.activate()
            check(self.cu.cuStreamDestroy(stream))

    def htod(self, dst: int, host, stream: int):
        check(self.cu.cuMemcpyHtoDAsync(dst, host.ctypes.data, host.nbytes, stream))

    def dtoh(self, host, src: int, stream: int):
        check(self.cu.cuMemcpyDtoHAsync(host.ctypes.data, src, host.nbytes, stream))

    def sync(self, stream: int):
        check(self.cu.cuStreamSynchronize(stream))

    def mem_info(self) -> dict:
        self.activate()
        free, total = check(self.cu.cuMemGetInfo())
        return {"free_mb": int(free // 2**20), "total_mb": int(total // 2**20)}
