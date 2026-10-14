"""Constant-velocity Kalman filter in (cx, cy, aspect, height) space.

State: [cx, cy, a, h, vcx, vcy, va, vh]. Process/measurement noise scale with the
box height (as in DeepSORT), so uncertainty grows with every predict() made
without a correcting update - i.e. during missed detections / occlusion.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

# 0.95 quantile of the chi-square distribution, indexed by degrees of freedom.
CHI2_95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877}


def xyxy_to_xyah(box) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    return np.array([x1 + w / 2, y1 + h / 2, w / h, h])


def xyah_to_xyxy(xyah) -> np.ndarray:
    cx, cy, a, h = np.asarray(xyah, dtype=np.float64)[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class KalmanFilter:
    ndim = 4

    def __init__(self, std_weight_position: float = 1 / 20, std_weight_velocity: float = 1 / 160):
        dt = 1.0
        self._F = np.eye(2 * self.ndim)
        for i in range(self.ndim):
            self._F[i, self.ndim + i] = dt
        self._H = np.eye(self.ndim, 2 * self.ndim)
        self._wp = std_weight_position
        self._wv = std_weight_velocity

    def initiate(self, measurement: np.ndarray):
        mean = np.r_[measurement, np.zeros(self.ndim)]
        h = measurement[3]
        std = [2 * self._wp * h, 2 * self._wp * h, 1e-2, 2 * self._wp * h,
               10 * self._wv * h, 10 * self._wv * h, 1e-5, 10 * self._wv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self._wp * h, self._wp * h, 1e-2, self._wp * h,
               self._wv * h, self._wv * h, 1e-5, self._wv * h]
        Q = np.diag(np.square(std))
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + Q
        return mean, cov

    def project(self, mean, cov, confidence: float = 1.0):
        h = mean[3]
        # Lower-confidence detections are trusted less (NSA-style scaling).
        scale = 1.0 + (1.0 - float(np.clip(confidence, 0.0, 1.0)))
        std = [self._wp * h * scale, self._wp * h * scale, 1e-1, self._wp * h * scale]
        R = np.diag(np.square(std))
        return self._H @ mean, self._H @ cov @ self._H.T + R

    def update(self, mean, cov, measurement, confidence: float = 1.0):
        proj_mean, proj_cov = self.project(mean, cov, confidence)
        chol, lower = scipy.linalg.cho_factor(proj_cov, lower=True, check_finite=False)
        gain = scipy.linalg.cho_solve((chol, lower), (cov @ self._H.T).T, check_finite=False).T
        innovation = measurement - proj_mean
        new_mean = mean + innovation @ gain.T
        new_cov = cov - gain @ proj_cov @ gain.T
        return new_mean, new_cov

    def gating_distance(self, mean, cov, measurements: np.ndarray, only_position: bool = False):
        """Squared Mahalanobis distance between the track and (N, 4) xyah measurements."""
        proj_mean, proj_cov = self.project(mean, cov)
        measurements = np.atleast_2d(measurements)
        if only_position:
            proj_mean, proj_cov = proj_mean[:2], proj_cov[:2, :2]
            measurements = measurements[:, :2]
        chol = np.linalg.cholesky(proj_cov)
        d = measurements - proj_mean
        z = scipy.linalg.solve_triangular(chol, d.T, lower=True, check_finite=False,
                                          overwrite_b=True)
        return np.sum(z * z, axis=0)
