"""Geometry, appearance distances and gated Hungarian assignment.

The Hungarian solver only proposes a minimum-cost assignment; every pair is then
re-checked against an explicit acceptance threshold, and pairs that were gated
(infinite cost) can never be accepted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

INF_COST = 1e5


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def max_coverage_by_others(boxes: np.ndarray) -> np.ndarray:
    """For each box, the largest fraction of its area covered by any other box."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    n = len(b)
    if n < 2:
        return np.zeros(n)
    ix1 = np.maximum(b[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(b[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(b[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-9)
    cov = inter / area[:, None]
    np.fill_diagonal(cov, 0.0)
    return cov.max(axis=1)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - cosine similarity for L2-normalized rows, clipped to [0, 1]."""
    a = np.asarray(a, dtype=np.float32).reshape(len(a), -1)
    b = np.asarray(b, dtype=np.float32).reshape(len(b), -1)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    return np.clip(1.0 - a @ b.T, 0.0, 1.0)


def gallery_distance(galleries: List[np.ndarray], queries: np.ndarray) -> np.ndarray:
    """Min cosine distance between each track gallery (K_i, D) and each query (M, D).

    Tracks with an empty gallery get distance NaN (= no appearance evidence).
    """
    out = np.full((len(galleries), len(queries)), np.nan, dtype=np.float32)
    if len(queries) == 0:
        return out
    for i, g in enumerate(galleries):
        if g is not None and len(g):
            out[i] = cosine_distance(g, queries).min(axis=0)
    return out


@dataclass
class AssignmentResult:
    matches: List[Tuple[int, int]] = field(default_factory=list)
    unmatched_rows: List[int] = field(default_factory=list)
    unmatched_cols: List[int] = field(default_factory=list)
    # cost of each accepted match and the margin to the next-best alternative
    costs: List[float] = field(default_factory=list)
    margins: List[float] = field(default_factory=list)


def linear_assignment(cost: np.ndarray, thresh: float) -> AssignmentResult:
    """Hungarian assignment with explicit acceptance threshold.

    ``cost`` entries >= INF_COST (or non-finite) are gated and never accepted.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n_rows, n_cols = cost.shape if cost.ndim == 2 else (0, 0)
    result = AssignmentResult()
    if n_rows == 0 or n_cols == 0:
        result.unmatched_rows = list(range(n_rows))
        result.unmatched_cols = list(range(n_cols))
        return result
    safe = np.where(np.isfinite(cost), np.minimum(cost, INF_COST), INF_COST)
    rows, cols = linear_sum_assignment(safe)
    matched_r, matched_c = set(), set()
    for r, c in zip(rows, cols):
        if safe[r, c] >= INF_COST or safe[r, c] > thresh:
            continue
        # Margin: how much worse the best alternative for this row or column is.
        row_alt = np.delete(safe[r], c)
        col_alt = np.delete(safe[:, c], r)
        alt = min(row_alt.min() if row_alt.size else INF_COST,
                  col_alt.min() if col_alt.size else INF_COST)
        result.matches.append((int(r), int(c)))
        result.costs.append(float(safe[r, c]))
        result.margins.append(float(alt - safe[r, c]))
        matched_r.add(r)
        matched_c.add(c)
    result.unmatched_rows = [r for r in range(n_rows) if r not in matched_r]
    result.unmatched_cols = [c for c in range(n_cols) if c not in matched_c]
    return result
