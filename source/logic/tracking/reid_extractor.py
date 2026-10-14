"""Crop extraction + quality assessment around a ReID backend.

The tracker receives, per detection, an embedding and two flags:
  * ``valid``    - an embedding was computed (crop large enough / inside frame)
  * ``reliable`` - the crop is good enough to be *added* to a track's gallery
                   (high score, not heavily covered by other boxes, not cut off
                   by the frame border)
Unreliable-but-valid embeddings may still be used for matching, never for memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from source.config.settings import REID, ReIDConfig, TrackerConfig, TRACKER
from source.logic.tracking.association import max_coverage_by_others


@dataclass
class AppearanceBatch:
    embeddings: np.ndarray  # (N, D) float32, zero rows where not valid
    valid: np.ndarray       # (N,) bool
    reliable: np.ndarray    # (N,) bool


def crop_boxes(frame: np.ndarray, boxes: np.ndarray, cfg: ReIDConfig = REID):
    """Return list of crops (None for invalid) clipped to the frame."""
    h, w = frame.shape[:2]
    crops = []
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float32):
        if not np.all(np.isfinite([x1, y1, x2, y2])):
            crops.append(None)
            continue
        xa, ya = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
        xb, yb = int(min(w, np.ceil(x2))), int(min(h, np.ceil(y2)))
        if yb - ya < cfg.min_crop_h or xb - xa < cfg.min_crop_w:
            crops.append(None)
            continue
        crops.append(frame[ya:yb, xa:xb])
    return crops


class ReIDExtractor:
    def __init__(self, backend, reid_cfg: ReIDConfig = REID, tracker_cfg: TrackerConfig = TRACKER):
        self.backend = backend
        self.reid_cfg = reid_cfg
        self.tracker_cfg = tracker_cfg

    def __call__(self, frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                 only: Optional[np.ndarray] = None) -> AppearanceBatch:
        n = len(boxes)
        emb = np.zeros((n, self.reid_cfg.embedding_dim), dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        reliable = np.zeros(n, dtype=bool)
        if n == 0:
            return AppearanceBatch(emb, valid, reliable)
        crops = crop_boxes(frame, boxes, self.reid_cfg)
        wanted = np.ones(n, dtype=bool) if only is None else np.asarray(only, dtype=bool)
        idx = [i for i, c in enumerate(crops) if c is not None and wanted[i]]
        if idx:
            emb[idx] = self.backend.extract_embeddings([crops[i] for i in idx])
            valid[idx] = True
        coverage = max_coverage_by_others(boxes)
        h, w = frame.shape[:2]
        m = self.tracker_cfg.feature_border_margin
        b = np.asarray(boxes, dtype=np.float32)
        truncated = (b[:, 0] <= m) | (b[:, 1] <= m) | (b[:, 2] >= w - 1 - m) | (b[:, 3] >= h - 1 - m)
        reliable = (valid & (scores >= self.tracker_cfg.feature_min_conf)
                    & (coverage <= self.tracker_cfg.feature_max_overlap) & ~truncated)
        return AppearanceBatch(emb, valid, reliable)
