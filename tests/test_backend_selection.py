"""Automatic ReID backend selection and runtime fallback.

Everything here is MOCKED (fake backends / patched dependency probes) and runs on any
machine. It proves the *selection logic*, not that TensorRT or a GPU works: those
are covered by the ``hardware``-marked tests in test_reid_backends.py.
"""
import logging

import numpy as np
import pytest

import source.logic.inference_manager as sel
from source.config.settings import REID
from source.logic.inference_manager import (Candidate, NoBackendAvailable, ReferenceEmbeddings,
                                            probe_crops, select_backend)
from source.models.reid.inference import ReIDBackend, ReIDModelError

_PROJ = np.random.default_rng(7).normal(size=(3 * 16 * 8, 512)).astype(np.float32)


def fake_features(batch: np.ndarray) -> np.ndarray:
    """Deterministic 'model': downsample + fixed random projection (like a tiny CNN)."""
    b = batch.shape[0]
    small = batch.reshape(b, 3, 16, batch.shape[2] // 16, 8, batch.shape[3] // 8).mean((3, 5))
    return small.reshape(b, -1) @ _PROJ


class FakeBackend(ReIDBackend):
    def __init__(self, name, device="cpu", noise=0.0, fail_infer=False, nan=False):
        super().__init__(REID)
        self.name, self.device, self.noise = name, device, noise
        self.fail_infer, self.nan, self.calls, self.closed = fail_infer, nan, 0, False

    def _infer(self, batch):
        self.calls += 1
        if self.fail_infer:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        out = fake_features(batch)
        if self.noise:  # systematic error (e.g. wrong weights/precision), batch-consistent
            bias = np.random.default_rng(0).normal(size=out.shape[1]).astype(np.float32)
            out = out + self.noise * np.abs(out).mean(axis=1, keepdims=True) * bias
        if self.nan:
            out[0, 0] = np.nan
        return out

    def close(self):
        self.closed = True


def reference():
    return ReferenceEmbeddings(REID, compute=lambda: ("pytorch-cpu", FakeBackend("ref")
                                                      .extract_embeddings(probe_crops())))


def cand(name, available=True, reason="", make=None, **kw):
    made = []

    def factory(cfg):
        if make is not None:
            b = make()
        else:
            b = FakeBackend(name, device="cpu" if name.endswith("cpu") else "cuda", **kw)
        made.append(b)
        return b

    c = Candidate(name, lambda: (available, reason), factory)
    c.made = made
    return c


def raises(exc):
    def make():
        raise exc
    return make


def names(result):
    return [(a["backend"], a["status"]) for a in result.attempts]


# ------------------------------------------------------------ selection order
def test_tensorrt_available_and_loads_is_selected():
    r = select_backend(REID, [cand("tensorrt"), cand("onnx-cuda"), cand("onnx-cpu")], reference())
    assert r.backend.name == "tensorrt"
    assert names(r) == [("tensorrt", "selected")]
    a = r.attempts[0]
    assert a["min_cosine_vs_reference"] > 0.99999 and a["reference"] == "pytorch-cpu"
    assert "init_s" in a and "warmup_s" in a


def test_tensorrt_package_missing_falls_back_to_next():
    r = select_backend(REID, [cand("tensorrt", False, "TensorRT Python package is not installed"),
                              cand("onnx-cuda")], reference())
    assert r.backend.name == "onnx-cuda"
    assert r.attempts[0] == {"backend": "tensorrt", "status": "unavailable",
                             "reason": "TensorRT Python package is not installed"}


def test_tensorrt_engine_incompatible_or_corrupted_falls_back():
    bad = cand("tensorrt", make=raises(ReIDModelError(
        "Failed to deserialize the TensorRT engine (corrupted, or built for a different GPU)")))
    r = select_backend(REID, [bad, cand("onnx-cpu")], reference())
    assert r.backend.name == "onnx-cpu"
    assert r.attempts[0]["status"] == "failed" and "deserialize" in r.attempts[0]["reason"]


def test_tensorrt_inference_failure_during_verification_falls_back():
    trt = cand("tensorrt", fail_infer=True)
    r = select_backend(REID, [trt, cand("onnx-cuda")], reference())
    assert r.backend.name == "onnx-cuda"
    assert "illegal memory access" in r.attempts[0]["reason"]
    assert trt.made[0].closed  # resources released


def test_cuda_onnxruntime_selected_when_tensorrt_unavailable():
    r = select_backend(REID, [cand("tensorrt", False, "no TensorRT"), cand("onnx-cuda"),
                              cand("pytorch-cuda"), cand("onnx-cpu")], reference())
    assert r.backend.name == "onnx-cuda" and r.backend.device == "cuda"


def test_cpu_only_environment_selects_onnx_cpu_and_warns(caplog):
    caplog.set_level(logging.WARNING, logger="ps2.reid.select")
    r = select_backend(REID, [cand("tensorrt", False, "no CUDA GPU"),
                              cand("onnx-cuda", False, "no CUDAExecutionProvider"),
                              cand("pytorch-cuda", False, "torch has no CUDA"),
                              cand("onnx-cpu"), cand("pytorch-cpu")], reference())
    assert r.backend.name == "onnx-cpu" and r.backend.device == "cpu"
    assert any("running on CPU" in m for m in caplog.messages)


def test_pytorch_only_environment():
    r = select_backend(REID, [cand("onnx-cpu", False, "onnxruntime is not installed"),
                              cand("pytorch-cpu")], reference())
    assert r.backend.name == "pytorch-cpu"


def test_no_compatible_backend_raises_with_every_reason():
    with pytest.raises(NoBackendAvailable) as ei:
        select_backend(REID, [cand("tensorrt", False, "no TensorRT"),
                              cand("onnx-cpu", False, "onnxruntime is not installed"),
                              cand("pytorch-cpu", make=raises(ReIDModelError("checkpoint missing")))],
                       reference())
    msg = str(ei.value)
    assert "no TensorRT" in msg and "onnxruntime is not installed" in msg and "checkpoint missing" in msg


# --------------------------------------------------------------- verification
def test_backend_with_wrong_numerics_is_rejected():
    r = select_backend(REID, [cand("tensorrt", noise=0.5), cand("onnx-cpu")], reference())
    assert r.backend.name == "onnx-cpu"
    assert "deviate" in r.attempts[0]["reason"]


def test_backend_with_nan_output_is_rejected():
    r = select_backend(REID, [cand("onnx-cuda", nan=True), cand("onnx-cpu")], reference())
    assert r.backend.name == "onnx-cpu" and "non-finite" in r.attempts[0]["reason"]


def test_small_numeric_differences_within_tolerance_are_accepted():
    r = select_backend(REID, [cand("tensorrt", noise=1e-4)], reference())
    assert r.backend.name == "tensorrt"
    assert REID.verify_min_cosine <= r.attempts[0]["min_cosine_vs_reference"] < 1.0


def test_reference_unavailable_still_checks_shape_and_finiteness():
    ref = ReferenceEmbeddings(REID, compute=lambda: (_ for _ in ()).throw(ReIDModelError("no ref")))
    r = select_backend(REID, [cand("onnx-cpu")], ref)
    assert r.backend.name == "onnx-cpu" and r.attempts[0]["reference"] is None
    assert "no ref" in r.attempts[0]["reference_note"]


# ----------------------------------------------------------- runtime fallback
def test_runtime_inference_failure_switches_to_next_verified_backend():
    trt = cand("tensorrt")
    r = select_backend(REID, [trt, cand("onnx-cuda"), cand("onnx-cpu")], reference())
    crops = probe_crops(3)
    expected = r.backend.extract_embeddings(crops)
    trt.made[0].fail_infer = True                   # e.g. GPU driver reset mid-video
    got = r.backend.extract_embeddings(crops)       # transparently retried on the fallback
    assert r.backend.name == "onnx-cuda"
    np.testing.assert_allclose(got, expected, atol=1e-5)  # same embeddings -> same tracking
    info = r.backend.info()
    assert info["fallback_events"][0]["from"] == "tensorrt"
    assert info["fallback_events"][0]["to"] == "onnx-cuda"
    assert trt.made[0].closed


def test_runtime_failure_without_fallback_raises_clear_error():
    trt = cand("tensorrt")
    r = select_backend(REID, [trt], reference())
    trt.made[0].fail_infer = True
    with pytest.raises(NoBackendAvailable, match="no fallback backend"):
        r.backend.extract_embeddings(probe_crops(2))


def test_backend_is_cached_by_registry(monkeypatch):
    from source.models.model_registry import ModelRegistry

    calls = []

    def fake_select(cfg, requested=None):
        calls.append(requested)
        return select_backend(REID, [cand("onnx-cpu")], reference())

    monkeypatch.setattr(sel, "select_backend", fake_select)
    reg = ModelRegistry()
    a, b = reg.reid(), reg.reid()
    assert a is b and calls == ["auto"]
    st = reg.status()
    assert st["reid_backend"]["backend"] == "onnx-cpu"
    assert st["reid_selection"]["selected"]["backend"] == "onnx-cpu"


# --------------------------------------------- real dependency probes (patched)
def _patch_env(monkeypatch, trt=None, cuda=False, ort=("CPUExecutionProvider",), torch_cuda=False,
               torch_ok=True):
    import source.utils.device_utils as gr

    monkeypatch.setattr(gr, "tensorrt_version", lambda: trt)
    monkeypatch.setattr(gr, "probe_cuda", lambda: {"available": cuda, "reason": "no CUDA device",
                                                   "name": "Fake GPU" if cuda else None})
    monkeypatch.setattr(sel, "_ort_providers",
                        lambda: (list(ort), "") if ort is not None else (None, "onnxruntime is not installed"))
    monkeypatch.setattr(sel, "_torch_cuda",
                        lambda: (True, "") if torch_cuda else (False, "torch has no usable CUDA device"))
    monkeypatch.setattr(sel, "_torch", lambda: (True, "") if torch_ok else (False, "torch is not installed"))


def _avail(monkeypatch, **kw):
    _patch_env(monkeypatch, **kw)
    return {c.name: c.available() for c in sel.default_candidates(REID)}


def test_probe_tensorrt_missing(monkeypatch):
    a = _avail(monkeypatch, trt=None, cuda=True)
    assert a["tensorrt"] == (False, "TensorRT Python package is not installed")


def test_probe_cuda_unavailable_disables_gpu_candidates(monkeypatch):
    a = _avail(monkeypatch, trt="11.4.0", cuda=False)
    assert not a["tensorrt"][0] and "no CUDA GPU" in a["tensorrt"][1]
    assert not a["onnx-cuda"][0] and not a["pytorch-cuda"][0]
    assert a["onnx-cpu"][0] is REID.onnx_path.is_file()


def test_probe_onnxruntime_unavailable(monkeypatch):
    a = _avail(monkeypatch, ort=None)
    assert a["onnx-cpu"] == (False, "onnxruntime is not installed")
    assert a["onnx-cuda"] == (False, "onnxruntime is not installed")
    assert a["pytorch-cpu"][0] is REID.checkpoint.is_file()


def test_probe_cuda_and_directml_providers(monkeypatch):
    a = _avail(monkeypatch, ort=("CUDAExecutionProvider", "DmlExecutionProvider",
                                 "CPUExecutionProvider"))
    if REID.onnx_path.is_file():
        assert a["onnx-cuda"] == (True, "") and a["onnx-dml"] == (True, "")


def test_probe_pytorch_cuda(monkeypatch):
    a = _avail(monkeypatch, torch_cuda=True)
    assert a["pytorch-cuda"][0] is REID.checkpoint.is_file()


def test_requested_backend_restricts_candidates():
    assert sel._resolve_order("auto") == sel.ORDER
    assert sel._resolve_order("onnx-cpu") == ("onnx-cpu",)
    assert sel._resolve_order("pytorch") == ("pytorch-cuda", "pytorch-cpu")
    with pytest.raises(ReIDModelError, match="Unknown ReID backend"):
        sel._resolve_order("openvino")
