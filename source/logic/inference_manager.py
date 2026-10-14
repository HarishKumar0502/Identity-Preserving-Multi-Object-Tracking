"""Automatic ReID backend/device selection with load + inference verification.

Order (first one that loads *and* passes verification wins):

    1. tensorrt      native TensorRT engine on an NVIDIA GPU
    2. onnx-cuda     ONNX Runtime CUDAExecutionProvider
    3. onnx-dml      ONNX Runtime DirectML (Windows GPUs)
    4. pytorch-cuda  PyTorch on CUDA
    5. onnx-cpu      ONNX Runtime CPU (measured faster than PyTorch CPU here)
    6. pytorch-cpu   PyTorch CPU (reference implementation)

Verification of a candidate: run a fixed, deterministic probe batch (batch 1 and 4)
and require (B, 512) finite, unit-norm embeddings that agree with the PyTorch CPU
reference (``ReIDConfig.verify_min_cosine``). A backend that loads but computes
something different is therefore rejected rather than silently changing ReID.

``ResilientBackend`` wraps the winner: if inference later raises (CUDA error, OOM,
driver reset), it switches to the next verified candidate and retries the batch,
recording the event; embeddings stay interchangeable because every candidate was
verified against the same reference.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from source.config.settings import REID, ReIDConfig
from source.models.model_registry import backend_factories
from source.models.reid.inference import ReIDBackend, ReIDModelError, l2_normalize

log = logging.getLogger("ps2.reid.select")

ORDER = ("tensorrt", "onnx-cuda", "onnx-dml", "pytorch-cuda", "onnx-cpu", "pytorch-cpu")
CPU_NAMES = {"onnx-cpu", "pytorch-cpu"}


class NoBackendAvailable(ReIDModelError):
    """No candidate could be loaded and verified."""


@dataclass
class Candidate:
    name: str
    available: Callable[[], Tuple[bool, str]]   # dependency check (cheap, no model load)
    factory: Callable[[ReIDConfig], ReIDBackend]


# ------------------------------------------------------------- availability
def _ort_providers() -> Tuple[Optional[List[str]], str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return None, "onnxruntime is not installed"
    return ort.get_available_providers(), ""


def _torch_cuda() -> Tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"
    if not torch.cuda.is_available():
        return False, f"torch {torch.__version__} has no usable CUDA device"
    return True, ""


def _torch() -> Tuple[bool, str]:
    try:
        import torch  # noqa: F401
        return True, ""
    except ImportError:
        return False, "torch is not installed"


def _availability(name: str, cfg: ReIDConfig) -> Callable[[], Tuple[bool, str]]:
    def onnx_file() -> Tuple[bool, str]:
        return (True, "") if cfg.onnx_path.is_file() else (
            False, f"ONNX model {cfg.onnx_path.name} not found (run scripts/export_reid_onnx.py)")

    def ckpt() -> Tuple[bool, str]:
        return (True, "") if cfg.checkpoint.is_file() else (
            False, f"checkpoint {cfg.checkpoint.name} not found")

    def ort_provider(provider: str):
        def check():
            providers, why = _ort_providers()
            if providers is None:
                return False, why
            if provider not in providers:
                return False, f"onnxruntime has no {provider} (available: {', '.join(providers)})"
            return onnx_file()
        return check

    def tensorrt():
        from source.utils.device_utils import probe_cuda, tensorrt_version
        if tensorrt_version() is None:
            return False, "TensorRT Python package is not installed"
        gpu = probe_cuda()
        if not gpu["available"]:
            return False, f"no CUDA GPU for TensorRT ({gpu['reason']})"
        return onnx_file()

    def both(*checks):
        def check():
            for c in checks:
                ok, why = c()
                if not ok:
                    return ok, why
            return True, ""
        return check

    return {
        "tensorrt": tensorrt,
        "onnx-cuda": ort_provider("CUDAExecutionProvider"),
        "onnx-dml": ort_provider("DmlExecutionProvider"),
        "pytorch-cuda": both(_torch_cuda, ckpt),
        "onnx-cpu": ort_provider("CPUExecutionProvider"),
        "pytorch-cpu": both(_torch, ckpt),
    }[name]


def default_candidates(cfg: ReIDConfig = REID) -> List[Candidate]:
    factories = backend_factories()
    return [Candidate(n, _availability(n, cfg), factories[n]) for n in ORDER]


# ------------------------------------------------------------- verification
def probe_crops(n: int = 4, seed: int = 1234) -> List[np.ndarray]:
    """Deterministic person-sized BGR crops (smooth structure + noise) of varying sizes."""
    rng = np.random.default_rng(seed)
    crops = []
    for i in range(n):
        h, w = int(rng.integers(96, 320)), int(rng.integers(40, 140))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        base = np.stack([128 + 100 * np.sin(yy / (7 + 3 * i) + c) * np.cos(xx / (5 + i))
                         for c in range(3)], axis=-1)
        crops.append(np.clip(base + rng.normal(0, 20, base.shape), 0, 255).astype(np.uint8))
    return crops


class ReferenceEmbeddings:
    """Lazily computes PyTorch-CPU (fallback: ONNX-CPU) embeddings for the probe crops."""

    def __init__(self, cfg: ReIDConfig, compute: Optional[Callable[[], Tuple[str, np.ndarray]]] = None):
        self.cfg = cfg
        self._compute = compute or self._default
        self._value: Optional[Tuple[str, np.ndarray]] = None
        self.error: Optional[str] = None

    def _default(self) -> Tuple[str, np.ndarray]:
        factories = backend_factories()
        errors = []
        for name in ("pytorch-cpu", "onnx-cpu"):
            ok, why = _availability(name, self.cfg)()
            if not ok:
                errors.append(f"{name}: {why}")
                continue
            try:
                b = factories[name](self.cfg)
                try:
                    return name, b.extract_embeddings(probe_crops())
                finally:
                    b.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")
        raise ReIDModelError("no CPU reference available (" + "; ".join(errors) + ")")

    def get(self) -> Optional[Tuple[str, np.ndarray]]:
        if self._value is None and self.error is None:
            try:
                self._value = self._compute()
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
                log.warning("ReID reference embeddings unavailable: %s", exc)
        return self._value

    def seed(self, name: str, emb: np.ndarray):
        if self._value is None:
            self._value = (name, emb)


def verify_backend(backend: ReIDBackend, cfg: ReIDConfig, reference: ReferenceEmbeddings) -> dict:
    """Run the probe batch; raise ReIDModelError if output is malformed or deviates."""
    crops = probe_crops()
    t0 = time.perf_counter()
    first = backend.extract_embeddings(crops[:1])   # warm-up + batch-1 path
    warm_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    emb = backend.extract_embeddings(crops)
    probe_s = time.perf_counter() - t0
    d = cfg.embedding_dim
    emb = np.asarray(emb)
    if emb.shape != (len(crops), d) or np.asarray(first).shape != (1, d):
        raise ReIDModelError(f"wrong embedding shape {emb.shape} (expected ({len(crops)}, {d}))")
    if not np.all(np.isfinite(emb)):
        raise ReIDModelError("non-finite values in embeddings")
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ReIDModelError(f"embeddings are not unit-norm (norms {norms.round(4).tolist()})")
    batch_cos = float(np.sum(l2_normalize(first)[0] * emb[0]))
    if batch_cos < cfg.verify_min_cosine:
        raise ReIDModelError(f"batch-1 and batch-4 results disagree (cos {batch_cos:.6f})")
    report = {"warmup_s": round(warm_s, 4), "probe_batch_s": round(probe_s, 4)}
    ref = reference.get()
    if ref is None:
        report["reference"] = None
        report["reference_note"] = f"not compared: {reference.error}"
        return report
    ref_name, ref_emb = ref
    cos = np.sum(ref_emb * emb, axis=1)
    report.update(reference=ref_name, min_cosine_vs_reference=round(float(cos.min()), 7),
                  max_abs_diff_vs_reference=float(np.abs(ref_emb - emb).max()))
    if float(cos.min()) < cfg.verify_min_cosine:
        raise ReIDModelError(f"embeddings deviate from the {ref_name} reference (min cosine "
                             f"{cos.min():.6f} < {cfg.verify_min_cosine})")
    return report


# ---------------------------------------------------------------- selection
@dataclass
class SelectionResult:
    backend: "ResilientBackend"
    attempts: List[dict] = field(default_factory=list)

    def report(self) -> dict:
        return {"selected": self.backend.info(), "attempts": self.attempts}


def _try_candidate(c: Candidate, cfg: ReIDConfig, reference: ReferenceEmbeddings
                   ) -> Tuple[Optional[ReIDBackend], dict]:
    ok, why = c.available()
    if not ok:
        return None, {"backend": c.name, "status": "unavailable", "reason": why}
    t0 = time.perf_counter()
    backend = None
    try:
        backend = c.factory(cfg)
        init_s = time.perf_counter() - t0
        check = verify_backend(backend, cfg, reference)
    except Exception as exc:  # noqa: BLE001 - any failure moves on to the next candidate
        if backend is not None:
            try:
                backend.close()
            except Exception:  # noqa: BLE001
                pass
        return None, {"backend": c.name, "status": "failed",
                      "reason": f"{type(exc).__name__}: {exc}",
                      "elapsed_s": round(time.perf_counter() - t0, 3)}
    attempt = {"backend": c.name, "status": "selected", "init_s": round(init_s, 3), **check}
    attempt.update({k: v for k, v in backend.info().items() if k != "backend"})
    if getattr(backend, "build_meta", None):
        attempt["engine_built_now"] = True
        attempt["engine_build_time_s"] = backend.build_meta.get("build_time_s")
    return backend, attempt


def _resolve_order(requested: str) -> Tuple[str, ...]:
    requested = (requested or "auto").lower()
    if requested == "auto":
        return ORDER
    legacy = {"onnx": ("onnx-cuda", "onnx-dml", "onnx-cpu"),
              "pytorch": ("pytorch-cuda", "pytorch-cpu")}
    if requested in legacy:
        return legacy[requested]
    if requested in ORDER:
        return (requested,)
    raise ReIDModelError(f"Unknown ReID backend '{requested}'. Use 'auto' or one of {list(ORDER)}.")


def select_backend(cfg: ReIDConfig = REID, candidates: Optional[Sequence[Candidate]] = None,
                   reference: Optional[ReferenceEmbeddings] = None,
                   requested: Optional[str] = None) -> SelectionResult:
    """Pick the first candidate that loads and verifies; wrap it with runtime fallback."""
    if candidates is None:
        order = _resolve_order(requested or cfg.backend)
        candidates = [c for c in default_candidates(cfg) if c.name in order]
        candidates.sort(key=lambda c: order.index(c.name))
    candidates = list(candidates)
    reference = reference or ReferenceEmbeddings(cfg)
    attempts = []
    for i, c in enumerate(candidates):
        backend, attempt = _try_candidate(c, cfg, reference)
        attempts.append(attempt)
        log.log(logging.INFO if backend else logging.WARNING, "ReID backend %s: %s%s", c.name,
                attempt["status"], f" ({attempt['reason']})" if "reason" in attempt else "")
        if backend is not None:
            resilient = ResilientBackend(backend, candidates[i + 1:], cfg, reference, attempts)
            info = resilient.info()
            skipped = attempts[:-1]
            if c.name in CPU_NAMES and any(a["backend"] not in CPU_NAMES for a in skipped):
                log.warning("ReID is running on CPU (%s); GPU backends were not usable: %s",
                            c.name, "; ".join(f"{a['backend']}: {a['reason']}" for a in skipped))
            log.info("ReID backend selected: %s on %s (precision %s, model %s, init %.2fs, "
                     "min cos vs %s %s)", info["backend"], info["device"], info["precision"],
                     info["model"], attempt["init_s"], attempt.get("reference"),
                     attempt.get("min_cosine_vs_reference"))
            return SelectionResult(resilient, attempts)
    raise NoBackendAvailable("No compatible ReID backend could be loaded: " + "; ".join(
        f"{a['backend']}: {a['reason']}" for a in attempts))


class ResilientBackend:
    """Selected backend + verified fallback on inference failure (same interface)."""

    def __init__(self, primary: ReIDBackend, fallbacks: Sequence[Candidate], cfg: ReIDConfig,
                 reference: ReferenceEmbeddings, attempts: List[dict]):
        self.active = primary
        self.cfg = cfg
        self._fallbacks = list(fallbacks)
        self._reference = reference
        self.attempts = attempts
        self.fallback_events: List[dict] = []
        self._lock = threading.Lock()

    # passthroughs used by the tracker / summaries
    @property
    def name(self) -> str:
        return self.active.name

    @property
    def device(self) -> str:
        return self.active.device

    def _switch(self, failed: ReIDBackend, exc: Exception):
        with self._lock:
            if self.active is not failed:   # another thread already switched
                return
            reason = f"{type(exc).__name__}: {exc}"
            log.error("ReID backend %s failed during inference (%s); trying fallback",
                      failed.name, reason)
            while self._fallbacks:
                c = self._fallbacks.pop(0)
                backend, attempt = _try_candidate(c, self.cfg, self._reference)
                attempt["trigger"] = f"runtime fallback after {failed.name} failure"
                self.attempts.append(attempt)
                if backend is not None:
                    self.fallback_events.append({"from": failed.name, "to": backend.name,
                                                 "reason": reason, "time": time.time()})
                    try:
                        failed.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self.active = backend
                    log.warning("ReID backend switched %s -> %s", failed.name, backend.name)
                    return
            raise NoBackendAvailable(f"ReID backend {failed.name} failed ({reason}) and no "
                                     "fallback backend could be loaded.") from exc

    def _call(self, method: str, arg):
        while True:
            backend = self.active
            try:
                return getattr(backend, method)(arg)
            except NoBackendAvailable:
                raise
            except Exception as exc:  # noqa: BLE001
                self._switch(backend, exc)

    def extract_embeddings(self, crops):
        return self._call("extract_embeddings", crops)

    def embed_preprocessed(self, batch):
        return self._call("embed_preprocessed", batch)

    def close(self):
        self.active.close()

    def info(self) -> dict:
        info = dict(self.active.info())
        if self.fallback_events:
            info["fallback_events"] = list(self.fallback_events)
        return info
