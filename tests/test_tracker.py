"""Synthetic lifecycle / association tests. These verify tracker logic only; they
do not say anything about real-world ReID accuracy."""
import numpy as np

from source.config.settings import TrackerConfig, baseline_tracker_config
from tests.conftest import noisy, unit
from source.logic.tracking.track_manager import Tracker, TrackState


def step(tracker, people, rng, sigma=0.02):
    """people: list of (box, identity_feature) visible this frame."""
    if not people:
        return tracker.update(np.zeros((0, 4)), np.zeros(0), np.zeros((0, 512), np.float32),
                              np.zeros(0, bool), np.zeros(0, bool))
    boxes = np.array([p[0] for p in people], dtype=float)
    feats = np.stack([noisy(p[1], rng, sigma) for p in people])
    return tracker.update(boxes, np.full(len(people), 0.9), feats,
                          np.ones(len(people), bool), np.ones(len(people), bool))


def box_at(x, y=100, w=40, h=100):
    return [x, y, x + w, y + h]


def ids(outputs):
    return {o.track_id for o in outputs if o.observed}


def test_track_initiation_and_confirmation(rng, identities):
    tr = Tracker(TrackerConfig(n_init=3))
    assert ids(step(tr, [(box_at(100), identities[0])], rng)) == set()
    assert ids(step(tr, [(box_at(102), identities[0])], rng)) == set()
    out = step(tr, [(box_at(104), identities[0])], rng)
    assert ids(out) == {1}
    assert tr.tracks[0].state == TrackState.CONFIRMED


def test_tentative_track_expires_on_miss(rng, identities):
    tr = Tracker(TrackerConfig(n_init=3))
    step(tr, [(box_at(100), identities[0])], rng)
    step(tr, [], rng)
    assert tr.tracks == []


def test_unmatched_detection_creates_new_track(rng, identities):
    tr = Tracker(TrackerConfig(n_init=1))
    step(tr, [(box_at(100), identities[0])], rng)
    out = step(tr, [(box_at(102), identities[0]), (box_at(500), identities[1])], rng)
    assert ids(out) == {1, 2}


def test_low_score_detection_does_not_start_track(rng, identities):
    tr = Tracker(TrackerConfig(n_init=1))
    tr.update(np.array([box_at(100)], float), np.array([0.3]))
    assert tr.tracks == []


def test_occluded_then_lost_then_expired(rng, identities):
    cfg = TrackerConfig(n_init=1, max_occluded_frames=3, track_buffer=5)
    tr = Tracker(cfg)
    step(tr, [(box_at(100), identities[0])], rng)
    step(tr, [], rng)
    assert tr.tracks[0].state == TrackState.OCCLUDED
    for _ in range(3):
        step(tr, [], rng)
    assert tr.tracks[0].state == TrackState.LOST
    for _ in range(5):
        step(tr, [], rng)
    assert tr.tracks == []


def test_single_miss_does_not_delete_confirmed_track(rng, identities):
    tr = Tracker(TrackerConfig(n_init=1))
    step(tr, [(box_at(100), identities[0])], rng)
    step(tr, [], rng)
    out = step(tr, [(box_at(104), identities[0])], rng)
    assert ids(out) == {1}


def test_reassociation_restores_original_id(rng, identities):
    cfg = TrackerConfig(n_init=3, max_occluded_frames=5, track_buffer=60)
    tr = Tracker(cfg)
    for k in range(10):
        step(tr, [(box_at(100 + 2 * k), identities[0]), (box_at(400), identities[1])], rng)
    assert tr.tracks[0].track_id == 1
    for _ in range(20):  # person 0 fully hidden long enough to become LOST
        step(tr, [(box_at(400), identities[1])], rng)
    assert next(t for t in tr.tracks if t.track_id == 1).state == TrackState.LOST
    out = step(tr, [(box_at(150), identities[0]), (box_at(400), identities[1])], rng)
    assert ids(out) == {1, 2}
    recovered = next(o for o in out if o.track_id == 1)
    assert recovered.state == "reassociated"
    assert tr.stats.recoveries == 1 and tr.stats.tracks_confirmed == 2


def test_lost_track_not_recovered_by_different_person(rng, identities):
    cfg = TrackerConfig(n_init=3, max_occluded_frames=5, track_buffer=60)
    tr = Tracker(cfg)
    for k in range(10):
        step(tr, [(box_at(100), identities[0])], rng)
    for _ in range(10):
        step(tr, [], rng)
    for _ in range(3):  # a different identity appears at the same spot
        out = step(tr, [(box_at(100), identities[2])], rng)
    assert ids(out) == {2}
    assert tr.stats.recoveries == 0


def test_ambiguous_lookalikes_are_not_recovered(rng):
    """Two lost tracks with near-identical appearance: a returning detection that
    matches both equally must not be assigned either identity."""
    base = unit(rng.normal(size=512))
    twin_a = unit(base + 0.03 * unit(rng.normal(size=512)))
    twin_b = unit(base + 0.03 * unit(rng.normal(size=512)))
    cfg = TrackerConfig(n_init=3, max_occluded_frames=3, track_buffer=60)
    tr = Tracker(cfg)
    for _ in range(8):
        step(tr, [(box_at(100), twin_a), (box_at(160), twin_b)], rng, sigma=0.0)
    for _ in range(8):
        step(tr, [], rng)
    lost_ids = {t.track_id for t in tr.tracks if t.state == TrackState.LOST}
    assert lost_ids == {1, 2}
    out = step(tr, [(box_at(130), unit(twin_a + twin_b))], rng, sigma=0.0)
    assert tr.stats.recoveries == 0
    assert tr.stats.ambiguous_rejections >= 1
    assert ids(out) == set()


def test_no_duplicate_active_ids_in_crowd(rng, identities):
    tr = Tracker(TrackerConfig(n_init=2))
    for k in range(40):
        people = [(box_at(100 + 30 * i + (k % 7), 100 + 3 * i), identities[i]) for i in range(5)
                  if not (i == 2 and 10 <= k < 30)]
        out = step(tr, people, rng)
        observed = [o.track_id for o in out if o.observed]
        assert len(observed) == len(set(observed))
        active = tr.active_ids()
        assert len(active) == len(set(active))


def test_crossing_people_keep_ids(rng, identities):
    """Two people walk through each other; appearance keeps the identities."""
    tr = Tracker(TrackerConfig(n_init=2))
    history = {}
    for k in range(40):
        a = (box_at(100 + 8 * k), identities[0])
        b = (box_at(420 - 8 * k), identities[1])
        out = step(tr, [a, b], rng)
        for o in out:
            if o.observed:
                cx = (o.box[0] + o.box[2]) / 2
                history.setdefault(o.track_id, []).append(cx)
    assert set(history) == {1, 2}
    # ID 1 started on the left, so it must keep moving right throughout.
    assert np.all(np.diff(history[1]) > 0) and np.all(np.diff(history[2]) < 0)


def test_unreliable_crop_does_not_update_gallery(rng, identities):
    tr = Tracker(TrackerConfig(n_init=1))
    feat = identities[0]
    tr.update(np.array([box_at(100)], float), np.array([0.9]), feat[None], np.ones(1, bool),
              np.ones(1, bool))
    n = len(tr.tracks[0].gallery)
    tr.update(np.array([box_at(102)], float), np.array([0.9]), identities[1][None],
              np.ones(1, bool), np.zeros(1, bool))
    assert len(tr.tracks[0].gallery) == n


def test_baseline_has_no_recovery(rng, identities):
    tr = Tracker(baseline_tracker_config())
    for _ in range(5):
        step(tr, [(box_at(100), identities[0])], rng)
    for _ in range(10):
        step(tr, [], rng)
    for _ in range(4):
        out = step(tr, [(box_at(100), identities[0])], rng)
    assert ids(out) == {2} and tr.stats.recoveries == 0


def test_noisy_appearance_does_not_fragment_clear_tracks(rng, identities):
    """Regression: uniformly poor ReID (e.g. blurred crops) must not veto matches
    that geometry makes obvious; the tracker should degrade towards IoU tracking."""
    tr = Tracker(TrackerConfig(n_init=3))
    for k in range(30):
        people = [(box_at(100 + 2 * k), identities[0]), (box_at(500 - 2 * k), identities[1])]
        out = step(tr, people, rng, sigma=0.08)  # pairwise cosine distance ~0.8
    assert ids(out) == {1, 2}
    assert tr.stats.tracks_confirmed == 2


def test_partial_crop_cannot_restore_identity_directly(rng, identities):
    cfg = TrackerConfig(n_init=3, max_occluded_frames=5, track_buffer=60)
    tr = Tracker(cfg)
    for k in range(10):
        step(tr, [(box_at(100), identities[0])], rng)
    for _ in range(10):
        step(tr, [], rng)
    feat = noisy(identities[0], rng)[None]
    tr.update(np.array([box_at(110)], float), np.array([0.9]), feat, np.ones(1, bool),
              np.zeros(1, bool))  # valid embedding, but unreliable (e.g. truncated)
    assert tr.stats.recoveries == 0
