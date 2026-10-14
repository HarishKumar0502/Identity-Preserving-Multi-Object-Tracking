import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def identities(rng):
    """Five well-separated synthetic identities in 512-d."""
    return [unit(rng.normal(size=512)) for _ in range(5)]


def noisy(feat, rng, sigma=0.02):
    return unit(feat + rng.normal(scale=sigma, size=feat.shape))
