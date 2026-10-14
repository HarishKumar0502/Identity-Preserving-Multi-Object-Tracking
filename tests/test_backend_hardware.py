"""REAL backend tests: every backend that is usable on this machine is loaded and run.

Marked ``hardware``: a backend whose runtime/GPU is missing is *skipped* (with the
reason), never mocked. Passing here is the evidence that TensorRT / GPU inference
works; tests/test_backend_selection.py only covers selection logic with fakes.

    python -m pytest tests -m hardware -v        # just these
"""
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from source.config.settings import DETECTOR, REID, TRACKER
from source.logic.inference_manager import default_candidates, select_backend
from source.models.model_registry import create_backend
from source.models.reid.inference import ReIDModelError

pytestmark = [pytest.mark.hardware,
              pytest.mark.skipif(not (REID.checkpoint.is_file() and REID.onnx_path.is_file()),
                                 reason="checkpoint or ONNX model missing")]

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "people-detection.mp4"
GPU_NAMES = ("tensorrt", "onnx-cuda", "onnx-dml", "pytorch-cuda")
# Justified tolerances: FP32 kernels differ only in summation order (observed
# max |diff| ~2e-7 on unit vectors); FP16 keeps ~3 significant digits. The tracker's
# tightest decision threshold is a cosine distance of 0.25 (reid_thresh), so a
# 1e-3 cosine change is < 0.5 % of it.
MIN_COS = {"fp32": 0.99999, "fp16": 0.999}


def _real_crops():
    """Person crops from the sample video (fixed frames), else structured synthetic crops."""
    if not SAMPLE.is_file() or not Path(DETECTOR.weights).is_file():
        from source.logic.inference_manager import probe_crops
        return probe_crops(24)
    from source.models.detector.inference import create_detector

    det = create_detector()
    cap = cv2.VideoCapture(str(SAMPLE))
    crops, idx = [], 0
    while len(crops) < 32:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % 10:
            continue
        d = det.detect(frame, idx)
        for (x1, y1, x2, y2), s in zip(d.boxes.astype(int), d.scores):
            if s >= 0.5 and y2 - y1 > 40:
                crops.append(frame[y1:y2, x1:x2].copy())
    cap.release()
    return crops


@pytest.fixture(scope="module")
def crops():
    return _real_crops()


@pytest.fixture(scope="module")
def reference(crops):
    ref = create_backend("pytorch-cpu")
    return ref.extract_embeddings(crops)


def _load_or_skip(name, cfg=REID):
    c = next(c for c in default_candidates(cfg) if c.name == name)
    ok, why = c.available()
    if not ok:
        pytest.skip(f"{name} not available on this machine: {why}")
    return c.factory(cfg)


@pytest.mark.parametrize("name", ["tensorrt", "onnx-cuda", "onnx-dml", "pytorch-cuda",
                                  "onnx-cpu", "pytorch-cpu"])
def test_backend_matches_pytorch_reference(name, crops, reference):
    backend = _load_or_skip(name)
    try:
        emb = backend.extract_embeddings(crops)
        assert emb.shape == (len(crops), REID.embedding_dim) and emb.dtype == np.float32
        assert np.isfinite(emb).all()
        np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)
        cos = (emb * reference).sum(1)
        assert cos.min() >= MIN_COS[backend.precision], (name, float(cos.min()))
        # ReID matching preserved: same nearest neighbour for every crop and
        # pairwise similarities essentially unchanged.
        s_ref, s_got = reference @ reference.T, emb @ emb.T
        np.fill_diagonal(s_ref, -9)
        np.fill_diagonal(s_got, -9)
        assert (s_ref.argmax(1) == s_got.argmax(1)).all()
        assert np.abs(s_ref - s_got).max() < 1e-3
        # batching invariance on this backend
        single = np.concatenate([backend.extract_embeddings([c]) for c in crops[:4]])
        np.testing.assert_allclose(single, emb[:4], atol=1e-4)
        if name in GPU_NAMES:
            assert "cuda" in backend.device or "directml" in backend.device
    finally:
        backend.close()


def test_tensorrt_fp16_engine_within_tolerance(crops, reference):
    cfg = replace(REID, tensorrt_precision="fp16")
    if not cfg.tensorrt_path.is_file():
        pytest.skip("FP16 engine not built (scripts/build_reid_tensorrt.py --precision fp16)")
    backend = _load_or_skip("tensorrt", replace(cfg, tensorrt_auto_build=False))
    try:
        cos = (backend.extract_embeddings(crops) * reference).sum(1)
        assert cos.min() >= MIN_COS["fp16"]
    finally:
        backend.close()


def test_tensorrt_corrupted_engine_is_rejected(tmp_path):
    from source.models.tensorrt.inference import TensorRTBackend
    from source.utils.device_utils import probe_cuda, tensorrt_version

    if tensorrt_version() is None or not probe_cuda()["available"]:
        pytest.skip("TensorRT/GPU not available")
    junk = tmp_path / "junk.engine"
    junk.write_bytes(b"\x00not an engine" * 1000)
    with pytest.raises(ReIDModelError, match="deserialize"):
        TensorRTBackend(REID, path=junk)


def test_tensorrt_stale_or_tampered_engine_detected(tmp_path):
    import json
    import shutil

    from source.utils.device_utils import tensorrt_version
    from source.models.tensorrt.engine_builder import engine_status, metadata_path

    if tensorrt_version() is None or not REID.tensorrt_path.is_file():
        pytest.skip("TensorRT engine not available")
    eng = tmp_path / REID.tensorrt_path.name
    shutil.copy(REID.tensorrt_path, eng)
    meta = json.loads(metadata_path(REID.tensorrt_path).read_text())
    args = (REID.onnx_path, REID.tensorrt_precision, REID.input_size, REID.tensorrt_max_batch,
            REID.tensorrt_opt_batch)
    metadata_path(eng).write_text(json.dumps(meta))
    assert engine_status(eng, *args) == (True, "ok")
    metadata_path(eng).write_text(json.dumps({**meta, "tensorrt_version": "10.0.0"}))
    ok, why = engine_status(eng, *args)
    assert not ok and "stale" in why and "tensorrt_version" in why
    metadata_path(eng).write_text(json.dumps(meta))
    with open(eng, "r+b") as f:  # flip bytes in the engine body
        f.seek(1000)
        f.write(b"\xff" * 16)
    ok, why = engine_status(eng, *args)
    assert not ok and "checksum" in why
    metadata_path(eng).unlink()
    assert engine_status(eng, *args)[0] is False


def test_auto_selection_on_this_machine_reports_a_verified_backend():
    result = select_backend(REID)
    try:
        sel = result.attempts[-1]
        assert sel["status"] == "selected"
        assert sel.get("reference") is not None
        assert sel["min_cosine_vs_reference"] >= REID.verify_min_cosine
        print("selected:", result.backend.info(), "attempts:",
              [(a["backend"], a["status"]) for a in result.attempts])
    finally:
        result.backend.close()


# --------------------------------------------- identity preservation (tracker)
class _ReplayDetector:
    name = "replay"

    def __init__(self, dets):
        self.dets = dets

    def detect(self, frame, frame_index):
        return self.dets[frame_index - 1]


@pytest.fixture(scope="module")
def recorded_detections():
    if not SAMPLE.is_file() or not Path(DETECTOR.weights).is_file():
        pytest.skip("sample video or detector weights missing")
    from source.models.detector.inference import create_detector

    det = create_detector()
    cap = cv2.VideoCapture(str(SAMPLE))
    dets = []
    for i in range(1, 241):
        ok, frame = cap.read()
        if not ok:
            break
        dets.append(det.detect(frame, i))
    cap.release()
    return dets


def _track_with(backend, dets, tmp_path, tag):
    from source.logic.video_processor import VideoProcessor

    clip = tmp_path / "clip.mp4"
    if not clip.exists():
        cap = cv2.VideoCapture(str(SAMPLE))
        w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (768, 432))
        for _ in range(len(dets)):
            ok, f = cap.read()
            w.write(f)
        w.release()
        cap.release()
    mot = tmp_path / f"{tag}.txt"
    s = VideoProcessor(_ReplayDetector(dets), backend, TRACKER, REID).process(
        clip, tmp_path / f"{tag}.mp4", mot)
    return s, mot.read_text().splitlines()


@pytest.mark.parametrize("name", ["tensorrt", "onnx-cpu"])
def test_backend_change_preserves_track_identities(name, recorded_detections, tmp_path):
    """Same detections, different ReID backend -> identical track IDs on every frame."""
    ref_s, ref_rows = _track_with(create_backend("pytorch-cpu"), recorded_detections, tmp_path, "ref")
    backend = _load_or_skip(name)
    try:
        s, rows = _track_with(backend, recorded_detections, tmp_path, name)
    finally:
        backend.close()
    assert ref_s["tracking"]["num_unique_track_ids"] >= 1
    assert s["tracking"]["unique_track_ids"] == ref_s["tracking"]["unique_track_ids"]
    assert s["tracking"]["tracks_recovered_after_occlusion"] == \
        ref_s["tracking"]["tracks_recovered_after_occlusion"]
    key = lambda r: tuple(r.split(",")[:2])  # (frame, id)
    assert [key(r) for r in rows] == [key(r) for r in ref_rows]
