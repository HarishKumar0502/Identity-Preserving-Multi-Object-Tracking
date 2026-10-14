import numpy as np

from source.logic.tracking.association import (INF_COST, cosine_distance, gallery_distance,
                                               iou_matrix, linear_assignment,
                                               max_coverage_by_others)
from source.logic.tracking.kalman_filter import CHI2_95, KalmanFilter, xyah_to_xyxy, xyxy_to_xyah


def test_box_conversion_roundtrip():
    box = np.array([10.0, 20.0, 50.0, 120.0])
    np.testing.assert_allclose(xyah_to_xyxy(xyxy_to_xyah(box)), box, atol=1e-9)


def test_iou_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 50 / 150, 0.0])
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_coverage_by_others():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [100, 100, 110, 110]])
    np.testing.assert_allclose(max_coverage_by_others(boxes), [0.5, 0.5, 0.0])


def test_cosine_and_gallery_distance(identities):
    a, b = identities[0], identities[1]
    d = cosine_distance(np.stack([a]), np.stack([a, b]))
    assert d[0, 0] < 1e-6 and d[0, 1] > 0.5
    g = gallery_distance([np.stack([a, b]), None], np.stack([b]))
    assert g[0, 0] < 1e-6 and np.isnan(g[1, 0])


def test_kalman_predict_and_correct():
    kf = KalmanFilter()
    mean, cov = kf.initiate(xyxy_to_xyah([100, 100, 140, 200]))
    # Move right by 5 px/frame; the filter should learn the velocity.
    for k in range(1, 15):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, xyxy_to_xyah([100 + 5 * k, 100, 140 + 5 * k, 200]))
    assert abs(mean[4] - 5.0) < 0.5
    pred, _ = kf.predict(mean, cov)
    assert abs(pred[0] - (120 + 5 * 15)) < 2.0


def test_uncertainty_grows_without_updates():
    kf = KalmanFilter()
    mean, cov = kf.initiate(xyxy_to_xyah([0, 0, 40, 100]))
    traces = []
    for _ in range(10):
        mean, cov = kf.predict(mean, cov)
        traces.append(np.trace(cov[:4, :4]))
    assert all(b > a for a, b in zip(traces, traces[1:]))


def test_motion_gating_rejects_far_measurement():
    kf = KalmanFilter()
    mean, cov = kf.initiate(xyxy_to_xyah([100, 100, 140, 200]))
    mean, cov = kf.predict(mean, cov)
    near = xyxy_to_xyah([102, 100, 142, 200])
    far = xyxy_to_xyah([400, 100, 440, 200])
    d = kf.gating_distance(mean, cov, np.stack([near, far]))
    assert d[0] < CHI2_95[4] < d[1]


def test_hungarian_threshold_and_unmatched():
    cost = np.array([[0.1, 0.9, INF_COST],
                     [0.8, 0.2, INF_COST]])
    res = linear_assignment(cost, thresh=0.5)
    assert sorted(res.matches) == [(0, 0), (1, 1)]
    assert res.unmatched_cols == [2] and res.unmatched_rows == []
    # Same assignment rejected by a stricter acceptance threshold.
    res = linear_assignment(cost, thresh=0.15)
    assert res.matches == [(0, 0)] and res.unmatched_rows == [1]


def test_hungarian_never_accepts_gated_pair():
    res = linear_assignment(np.array([[INF_COST]]), thresh=1e9)
    assert res.matches == [] and res.unmatched_rows == [0] and res.unmatched_cols == [0]


def test_hungarian_empty_inputs():
    res = linear_assignment(np.zeros((0, 3)), 0.5)
    assert res.unmatched_cols == [0, 1, 2]
