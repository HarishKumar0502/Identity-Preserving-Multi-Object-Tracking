"""Checkpoint loading and backend-equivalence tests (need the PS-1 checkpoint)."""
import numpy as np
import pytest

from source.config.settings import REID, ReIDConfig
from source.logic.tracking.reid_extractor import ReIDExtractor, crop_boxes
from source.models.model_registry import create_backend
from source.models.reid.inference import (ReIDModelError, build_reid_model, load_checkpoint_state,
                                          preprocess_crops)

pytestmark = pytest.mark.skipif(not REID.checkpoint.is_file(), reason="PS-1 checkpoint not found")


@pytest.fixture(scope="module")
def torch_backend():
    return create_backend("pytorch")


@pytest.fixture(scope="module")
def crops():
    rng = np.random.default_rng(1)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for h, w in [(200, 80), (64, 30), (300, 140)]]


def test_checkpoint_is_resnet18_with_751_class_head():
    sd = load_checkpoint_state(REID.checkpoint)
    assert sd["fc.weight"].shape == (751, 512)
    assert sd["conv1.weight"].shape == (64, 3, 7, 7)


def test_model_strict_load_and_classifier_removed():
    import torch

    model = build_reid_model(REID)
    assert isinstance(model.fc, torch.nn.Identity)
    assert not model.training
    # Backbone weights really came from the checkpoint (not random init).
    sd = load_checkpoint_state(REID.checkpoint)
    assert torch.equal(model.layer4[1].conv2.weight, sd["layer4.1.conv2.weight"])


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(ReIDModelError):
        build_reid_model(ReIDConfig(checkpoint=tmp_path / "nope.pth"))


def test_incompatible_checkpoint_raises(tmp_path):
    import torch

    bad = tmp_path / "bad.pth"
    torch.save({"model_state_dict": {"foo.weight": torch.zeros(3)}}, bad)
    with pytest.raises(ReIDModelError, match="incompatible"):
        build_reid_model(ReIDConfig(checkpoint=bad))


def test_preprocessing_shape_and_normalization(crops):
    x = preprocess_crops(crops)
    assert x.shape == (3, 3, *REID.input_size) and x.dtype == np.float32
    assert -2.2 < x.min() and x.max() < 2.7  # ImageNet-normalized range


def test_embedding_shape_norm_and_determinism(torch_backend, crops):
    e1 = torch_backend.extract_embeddings(crops)
    e2 = torch_backend.extract_embeddings(crops)
    assert e1.shape == (3, 512)
    np.testing.assert_allclose(np.linalg.norm(e1, axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(e1, e2)
    assert torch_backend.extract_embeddings([]).shape == (0, 512)


def test_batching_does_not_change_embeddings(torch_backend, crops):
    together = torch_backend.extract_embeddings(crops)
    single = np.concatenate([torch_backend.extract_embeddings([c]) for c in crops])
    np.testing.assert_allclose(together, single, atol=1e-5)


def test_invalid_crops_are_skipped(torch_backend):
    frame = np.zeros((240, 320, 3), np.uint8)
    boxes = np.array([[10, 10, 60, 150], [100, 100, 103, 104], [315, 235, 400, 300],
                      [np.nan, 0, 10, 10]], dtype=np.float32)
    crops = crop_boxes(frame, boxes)
    assert crops[0] is not None and crops[1] is None and crops[2] is None and crops[3] is None
    app = ReIDExtractor(torch_backend)(frame, boxes, np.array([0.9, 0.9, 0.9, 0.9]))
    assert app.valid.tolist() == [True, False, False, False]
    assert not app.embeddings[1:].any()


def test_occluded_crop_marked_unreliable(torch_backend):
    frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    boxes = np.array([[10, 10, 60, 150], [20, 10, 70, 150], [200, 10, 250, 150]], np.float32)
    app = ReIDExtractor(torch_backend)(frame, boxes, np.array([0.9, 0.9, 0.9]))
    assert app.valid.all()
    assert app.reliable.tolist() == [False, False, True]


def test_border_truncated_crop_marked_unreliable(torch_backend):
    frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    boxes = np.array([[100, 50, 150, 200], [200, 120, 250, 239], [0, 50, 40, 200]], np.float32)
    app = ReIDExtractor(torch_backend)(frame, boxes, np.array([0.9, 0.9, 0.9]))
    assert app.valid.all()
    assert app.reliable.tolist() == [True, False, False]
