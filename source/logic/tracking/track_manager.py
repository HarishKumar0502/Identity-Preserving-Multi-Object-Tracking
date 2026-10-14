"""Online, causal multi-object tracker with explicit lifecycle and ID recovery.

Per-frame pipeline (``Tracker.update``):

1. Kalman-predict every live track (uncertainty grows while unobserved).
2. Stage 1  - confirmed/occluded tracks  x high-score detections:
              fused motion (Mahalanobis) + appearance (ReID) + IoU cost, hard-gated.
3. Stage 2  - still-unmatched active tracks x low-score detections: IoU only
              (low scores are usually partially occluded people; their crops are
              never used to update appearance memory).
4. Stage 3  - tentative tracks x remaining high-score detections: IoU (+ app gate).
5. Stage 4  - lost tracks x remaining high-score detections: appearance-first
              re-association with a growing spatial window, strict threshold and
              a ratio test against the second-best candidate (ambiguity -> reject).
6. New tentative tracks from leftover confident detections (duplicate-suppressed).
   On confirmation a tentative track gets one more, multi-frame-evidence attempt
   to recover a lost identity before a fresh ID is issued.
7. Lifecycle bookkeeping for unmatched tracks (occluded -> lost -> expired).

States: tentative, confirmed, occluded, lost, reassociated, expired.
Display IDs are issued only on confirmation, are unique for the tracker's
lifetime, and a recovered track keeps its original ID.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional

import numpy as np

from source.config.settings import TRACKER, TrackerConfig
from source.logic.tracking.association import (INF_COST, cosine_distance, gallery_distance,
                                               iou_matrix, linear_assignment)
from source.logic.tracking.kalman_filter import KalmanFilter, xyah_to_xyxy, xyxy_to_xyah


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    OCCLUDED = "occluded"
    LOST = "lost"
    REASSOCIATED = "reassociated"
    EXPIRED = "expired"


ACTIVE_STATES = (TrackState.CONFIRMED, TrackState.OCCLUDED, TrackState.REASSOCIATED)


class Track:
    _uid_counter = 0

    def __init__(self, box, score, kf: KalmanFilter, cfg: TrackerConfig, frame_idx: int,
                 feature: Optional[np.ndarray] = None):
        Track._uid_counter += 1
        self.uid = Track._uid_counter
        self.track_id: Optional[int] = None
        self.cfg = cfg
        self.kf = kf
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(box))
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.start_frame = frame_idx
        self.last_box = np.asarray(box, dtype=np.float64)
        self.last_score = float(score)
        self.lost_frame: Optional[int] = None
        self.gallery: Deque[np.ndarray] = deque(maxlen=cfg.gallery_size)
        self.smooth_feat: Optional[np.ndarray] = None
        # Short-term memory: latest valid embedding of any quality. Used only for
        # frame-to-frame matching, never for identity recovery.
        self.recent_feat: Optional[np.ndarray] = None
        self.n_recoveries = 0
        self.n_occlusions = 0
        if feature is not None:
            self.add_feature(feature, force=True)

    # ---------------------------------------------------------------- motion
    @property
    def predicted_box(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])

    def predict(self):
        if self.mean[3] + self.mean[7] <= 1:  # keep height positive
            self.mean[7] = 0.0
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    # ------------------------------------------------------------ appearance
    def appearance_bank(self, short_term: bool = False) -> Optional[np.ndarray]:
        """Long-term identity memory (reliable crops only); ``short_term`` adds the
        latest observation, for matching an active track to the next frame."""
        feats = list(self.gallery)
        if self.smooth_feat is not None:
            feats.append(self.smooth_feat)
        if short_term and self.recent_feat is not None:
            feats.append(self.recent_feat)
        return np.stack(feats) if feats else None

    def add_feature(self, feat: np.ndarray, force: bool = False) -> bool:
        """Add a reliable feature unless it contradicts the track's identity."""
        feat = np.asarray(feat, dtype=np.float32)
        if self.smooth_feat is not None and not force:
            if float(cosine_distance(self.smooth_feat[None], feat[None])[0, 0]) > \
                    self.cfg.feature_outlier_dist:
                return False
        self.gallery.append(feat)
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            a = self.cfg.ema_alpha
            s = a * self.smooth_feat + (1 - a) * feat
            self.smooth_feat = s / max(np.linalg.norm(s), 1e-12)
        return True

    # ------------------------------------------------------------- lifecycle
    def update(self, box, score, frame_idx: int):
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(box), score)
        self.last_box = np.asarray(box, dtype=np.float64)
        self.last_score = float(score)
        self.hits += 1
        self.time_since_update = 0
        if self.state in (TrackState.OCCLUDED, TrackState.REASSOCIATED):
            self.state = TrackState.CONFIRMED

    def mark_lost(self, frame_idx: int):
        self.state = TrackState.LOST
        self.lost_frame = frame_idx
        self.recent_feat = None
        self.mean[4:] = 0.0  # stop extrapolating; position becomes a search anchor


@dataclass
class TrackOutput:
    track_id: int
    box: np.ndarray
    score: float
    state: str
    observed: bool  # False -> predicted box of an occluded track (render only)


@dataclass
class TrackerStats:
    tracks_confirmed: int = 0
    recoveries: int = 0
    recoveries_on_confirmation: int = 0
    occlusion_events: int = 0
    ambiguous_rejections: int = 0
    duplicate_suppressions: int = 0
    expired: int = 0
    recovered_ids: set = field(default_factory=set)


class Tracker:
    def __init__(self, cfg: TrackerConfig = TRACKER):
        self.cfg = cfg
        self.kf = KalmanFilter()
        self.tracks: List[Track] = []
        self.frame_idx = 0
        self._next_id = 1
        self.stats = TrackerStats()
        self._matched: set = set()

    # ----------------------------------------------------------------- utils
    def _tracks_in(self, *states) -> List[Track]:
        return [t for t in self.tracks if t.state in states]

    def active_ids(self) -> List[int]:
        return [t.track_id for t in self.tracks if t.state in ACTIVE_STATES]

    def _issue_id(self, track: Track):
        track.track_id = self._next_id
        self._next_id += 1
        self.stats.tracks_confirmed += 1

    # ------------------------------------------------------------ cost terms
    def _active_cost(self, tracks: List[Track], boxes, scores, emb, valid, reliable):
        """Fused, normalized, hard-gated cost between active tracks and detections."""
        cfg = self.cfg
        n, m = len(tracks), len(boxes)
        cost = np.full((n, m), INF_COST)
        if n == 0 or m == 0:
            return cost
        ious = iou_matrix(np.stack([t.predicted_box for t in tracks]), boxes)
        meas = np.stack([xyxy_to_xyah(b) for b in boxes])
        app = (gallery_distance([t.appearance_bank(short_term=True) for t in tracks], emb)
               if cfg.use_appearance else np.full((n, m), np.nan))
        for i, t in enumerate(tracks):
            maha = self.kf.gating_distance(t.mean, t.cov, meas)
            for j in range(m):
                iou = ious[i, j]
                # Motion gate: implausible jump unless boxes still overlap substantially.
                if maha[j] > cfg.chi2_gate and iou < 0.3:
                    continue
                has_app = cfg.use_appearance and valid[j] and not np.isnan(app[i, j])
                w_a = cfg.w_appearance if has_app else 0.0
                if has_app:
                    gate = cfg.appearance_gate if reliable[j] else cfg.appearance_gate + 0.15
                    # Appearance alone vetoes only when geometry is weak; with strong
                    # overlap a noisy embedding must not break an obvious continuation.
                    if app[i, j] > (cfg.appearance_veto if iou >= cfg.iou_trust else gate):
                        continue
                    if not reliable[j]:
                        w_a *= 0.5  # partial / overlapped crop -> weaker evidence
                elif iou < cfg.min_iou_without_appearance:
                    continue  # no appearance evidence and no overlap -> reject
                w_sum = cfg.w_motion + cfg.w_iou + w_a
                if w_sum <= 0:
                    continue
                # Strong overlap already validates the motion; the constant-velocity
                # model is unreliable for abrupt turns and border-truncated boxes.
                motion = 0.0 if iou >= cfg.iou_trust else min(maha[j] / cfg.chi2_gate, 1.0)
                c = (cfg.w_motion * motion
                     + cfg.w_iou * (1.0 - iou)
                     + (w_a * app[i, j] if has_app else 0.0)) / w_sum
                cost[i, j] = c
        return cost

    def _iou_cost(self, tracks: List[Track], boxes, emb=None, valid=None, thresh=1.0):
        n, m = len(tracks), len(boxes)
        if n == 0 or m == 0:
            return np.full((n, m), INF_COST)
        ious = iou_matrix(np.stack([t.predicted_box for t in tracks]), boxes)
        cost = 1.0 - ious
        cost[cost > thresh] = INF_COST
        if emb is not None and self.cfg.use_appearance:
            app = gallery_distance([t.appearance_bank(short_term=True) for t in tracks], emb)
            limit = np.where(ious >= self.cfg.iou_trust, self.cfg.appearance_veto,
                             self.cfg.appearance_gate)
            bad = (~np.isnan(app)) & valid[None, :] & (app > limit)
            cost[bad] = INF_COST
        return cost

    def _reassoc_cost(self, tracks: List[Track], boxes, emb, valid, reliable):
        """Appearance-first cost for lost tracks with a spatial window that grows
        with time lost. Returns (cost, raw appearance distance)."""
        cfg = self.cfg
        n, m = len(tracks), len(boxes)
        cost = np.full((n, m), INF_COST)
        app = gallery_distance([t.appearance_bank() for t in tracks], emb)
        det_c = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
        det_h = boxes[:, 3] - boxes[:, 1]
        for i, t in enumerate(tracks):
            if t.appearance_bank() is None:
                continue  # never recover identity without appearance history
            lb = t.last_box
            h = max(lb[3] - lb[1], 1.0)
            lost_for = self.frame_idx - (t.lost_frame or self.frame_idx) + cfg.max_occluded_frames
            radius = min(cfg.reassoc_radius + cfg.reassoc_radius_growth * lost_for,
                         cfg.reassoc_radius_max) * h
            anchor = np.array([(lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2])
            dist = np.linalg.norm(det_c - anchor, axis=1)
            for j in range(m):
                # Only full, unoccluded crops may restore an identity directly; partial
                # ones must wait for multi-frame evidence at confirmation time.
                if not (valid[j] and reliable[j]) or np.isnan(app[i, j]):
                    continue
                if dist[j] > radius or not (0.5 <= det_h[j] / h <= 2.0):
                    continue
                if app[i, j] > cfg.reid_thresh:
                    continue
                cost[i, j] = app[i, j] + 0.05 * dist[j] / radius
        return cost, app

    def _passes_ratio_test(self, app: np.ndarray, r: int, c: int) -> bool:
        best = app[r, c]
        others = np.concatenate([np.delete(app[r], c), np.delete(app[:, c], r)])
        others = others[~np.isnan(others)]
        if others.size == 0:
            return True
        return best <= self.cfg.ambiguity_ratio * others.min()

    # --------------------------------------------------------------- update
    def update(self, boxes: np.ndarray, scores: np.ndarray, embeddings: Optional[np.ndarray] = None,
               valid: Optional[np.ndarray] = None, reliable: Optional[np.ndarray] = None
               ) -> List[TrackOutput]:
        cfg = self.cfg
        self.frame_idx += 1
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        m = len(boxes)
        if embeddings is None:
            embeddings = np.zeros((m, 1), np.float32)
            valid = np.zeros(m, bool)
            reliable = np.zeros(m, bool)
        valid = np.asarray(valid if valid is not None else np.ones(m, bool), dtype=bool)
        reliable = np.asarray(reliable if reliable is not None else valid, dtype=bool)

        for t in self.tracks:
            t.predict()

        high = np.where(scores >= cfg.high_conf)[0]
        low = np.where(scores < cfg.high_conf)[0]
        used = np.zeros(m, dtype=bool)
        matched_tracks: set = set()
        self._matched = matched_tracks

        def apply(track: Track, j: int, allow_feature: bool):
            track.update(boxes[j], scores[j], self.frame_idx)
            used[j] = True
            matched_tracks.add(track.uid)
            if cfg.use_appearance and valid[j]:
                track.recent_feat = embeddings[j].astype(np.float32)
            if allow_feature and cfg.use_appearance and reliable[j]:
                track.add_feature(embeddings[j])

        # Stage 1: active tracks x high-score detections.
        active = self._tracks_in(*ACTIVE_STATES)
        cost = self._active_cost(active, boxes[high], scores[high], embeddings[high],
                                 valid[high], reliable[high])
        res = linear_assignment(cost, cfg.match_thresh)
        for (r, c), margin in zip(res.matches, res.margins):
            apply(active[r], high[c], allow_feature=margin >= cfg.ambiguous_margin)

        # Stage 2: remaining active tracks x low-score detections (IoU only).
        if cfg.low_conf_iou_thresh > 0 and len(low):
            rest = [t for t in active if t.uid not in matched_tracks]
            cost = self._iou_cost(rest, boxes[low], thresh=cfg.low_conf_iou_thresh)
            for r, c in linear_assignment(cost, cfg.low_conf_iou_thresh).matches:
                apply(rest[r], low[c], allow_feature=False)

        # Stage 3: tentative tracks x remaining high-score detections.
        free_high = high[~used[high]]
        tentative = self._tracks_in(TrackState.TENTATIVE)
        cost = self._iou_cost(tentative, boxes[free_high], embeddings[free_high],
                              valid[free_high], thresh=cfg.tentative_iou_thresh)
        for r, c in linear_assignment(cost, cfg.tentative_iou_thresh).matches:
            t = tentative[r]
            apply(t, free_high[c], allow_feature=True)
            if t.hits >= cfg.n_init:
                self._confirm(t)

        # Stage 4: lost tracks x remaining high-score detections (re-association).
        free_high = high[~used[high]]
        lost = self._tracks_in(TrackState.LOST)
        if cfg.enable_reassociation and cfg.use_appearance and lost and len(free_high):
            cost, app = self._reassoc_cost(lost, boxes[free_high], embeddings[free_high],
                                           valid[free_high], reliable[free_high])
            for r, c in linear_assignment(cost, cfg.reid_thresh + 0.05).matches:
                if not self._passes_ratio_test(app, r, c):
                    self.stats.ambiguous_rejections += 1
                    continue
                t = lost[r]
                apply(t, free_high[c], allow_feature=reliable[free_high[c]])
                self._recover(t)

        # Step 6: new tentative tracks (duplicate-suppressed).
        live_boxes = [t.last_box for t in self.tracks if t.uid in matched_tracks]
        for j in high[~used[high]]:
            if scores[j] < cfg.new_track_conf:
                continue
            if live_boxes and iou_matrix(boxes[j:j + 1], np.stack(live_boxes)).max() >= \
                    cfg.duplicate_iou:
                self.stats.duplicate_suppressions += 1
                continue
            feat = embeddings[j] if (cfg.use_appearance and reliable[j]) else None
            t = Track(boxes[j], scores[j], self.kf, cfg, self.frame_idx, feat)
            if cfg.use_appearance and valid[j]:
                t.recent_feat = embeddings[j].astype(np.float32)
            self.tracks.append(t)
            matched_tracks.add(t.uid)
            live_boxes.append(t.last_box)
            if cfg.n_init <= 1:
                self._confirm(t)

        # Step 7: lifecycle for unmatched tracks.
        for t in self.tracks:
            if t.uid in matched_tracks:
                continue
            if t.state == TrackState.TENTATIVE:
                t.state = TrackState.EXPIRED
            elif t.state in ACTIVE_STATES:
                if t.time_since_update > cfg.max_occluded_frames:
                    t.mark_lost(self.frame_idx)
                    if cfg.track_buffer <= 0:
                        t.state = TrackState.EXPIRED
                elif t.state != TrackState.OCCLUDED:
                    t.state = TrackState.OCCLUDED
                    t.n_occlusions += 1
                    self.stats.occlusion_events += 1
            elif t.state == TrackState.LOST:
                if t.time_since_update > cfg.max_occluded_frames + cfg.track_buffer:
                    t.state = TrackState.EXPIRED
        self._prune()
        return self._outputs()

    # ------------------------------------------------------------ recovery
    def _recover(self, t: Track):
        t.state = TrackState.REASSOCIATED
        t.n_recoveries += 1
        self.stats.recoveries += 1
        self.stats.recovered_ids.add(t.track_id)

    def _confirm(self, t: Track):
        """Promote a tentative track; first try to hand it a lost identity using
        the appearance evidence accumulated over its tentative frames."""
        if self.cfg.enable_reassociation and self.cfg.use_appearance and t.gallery:
            lost = self._tracks_in(TrackState.LOST)
            if lost:
                query = np.stack(list(t.gallery))
                query_mean = query.mean(0)
                query_mean /= max(np.linalg.norm(query_mean), 1e-12)
                cost, app = self._reassoc_cost(lost, t.last_box[None].astype(np.float64),
                                               query_mean[None], np.ones(1, bool),
                                               np.ones(1, bool))
                best = int(np.argmin(cost[:, 0]))
                if cost[best, 0] < INF_COST:
                    others = np.delete(app[:, 0], best)
                    others = others[~np.isnan(others)]
                    if others.size == 0 or app[best, 0] <= self.cfg.ambiguity_ratio * others.min():
                        self._absorb(lost[best], t)
                        return
                    self.stats.ambiguous_rejections += 1
        t.state = TrackState.CONFIRMED
        self._issue_id(t)

    def _absorb(self, old: Track, new: Track):
        """``old`` (lost) takes over ``new``'s motion state; ``new`` is discarded."""
        old.mean, old.cov = new.mean, new.cov
        old.last_box, old.last_score = new.last_box, new.last_score
        old.recent_feat = new.recent_feat
        old.time_since_update = 0
        self._matched.add(old.uid)
        old.hits += new.hits
        for f in new.gallery:
            old.add_feature(f)
        self._recover(old)
        self.stats.recoveries_on_confirmation += 1
        new.state = TrackState.EXPIRED

    def _prune(self):
        before = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.state != TrackState.EXPIRED]
        lost = [t for t in self.tracks if t.state == TrackState.LOST]
        if len(lost) > self.cfg.max_lost_tracks:
            drop = {t.uid for t in sorted(lost, key=lambda t: t.lost_frame)[
                :len(lost) - self.cfg.max_lost_tracks]}
            self.tracks = [t for t in self.tracks if t.uid not in drop]
        self.stats.expired += before - len(self.tracks)

    def _outputs(self) -> List[TrackOutput]:
        out = []
        for t in self.tracks:
            if t.track_id is None:
                continue
            if t.state in (TrackState.CONFIRMED, TrackState.REASSOCIATED) and t.time_since_update == 0:
                out.append(TrackOutput(t.track_id, t.last_box.copy(), t.last_score,
                                       t.state.value, True))
            elif t.state == TrackState.OCCLUDED and self.cfg.draw_predicted_occluded:
                out.append(TrackOutput(t.track_id, t.predicted_box, t.last_score,
                                       t.state.value, False))
        return out

    def state_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in self.tracks:
            counts[t.state.value] = counts.get(t.state.value, 0) + 1
        return counts
