"""
ngperception.tracking.trackers.sort
===================================

SORT — Simple Online and Realtime Tracking (Bewley et al., ICIP 2016).
The **basic** tier of the tracking task: the baseline every later method is measured
against, and the one that shows how far geometry alone gets you.

Two components, both classical:

1. a **constant-velocity Kalman filter** per track, on the state
   ``[cx, cy, s, r, vx, vy, vs]`` where ``s = w*h`` (area) and ``r = w/h`` (aspect).
   Aspect is treated as constant — SORT's own simplification, kept here so the
   baseline is the published one rather than a private variant;
2. **Hungarian assignment** (``scipy.optimize.linear_sum_assignment``) on the IoU
   between each track's predicted box and each detection, with matches below
   ``iou_threshold`` rejected after the assignment rather than before, so the
   assignment stays globally optimal.

No network, no weights, no GPU: numpy and scipy only. That is deliberate — it means
the tracking arm of the suite can be run and reproduced by anyone, and a later
appearance-based backend has a like-for-like reference to beat.

Known limits of the baseline, stated so the numbers are not over-read: SORT has no
re-identification, so an object that leaves and returns gets a new id and the
sequence takes an ID-switch; it has no occlusion handling beyond ``max_age`` coasting;
and it inherits every miss of the detector it is given.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .base import BaseTracker, TrackResult, iou_matrix, register


class _KalmanBoxTracker:
    """One tracked object: a constant-velocity Kalman filter over ``[cx, cy, s, r]``."""

    def __init__(self, box: np.ndarray, score: float, label: int, name: str, track_id: int):
        self.id = int(track_id)
        self.score = float(score)
        self.label = int(label)
        self.name = str(name)

        self.time_since_update = 0
        self.hits = 1
        self.age = 0

        # State transition: position advances by its velocity, aspect is constant.
        self._F = np.eye(7, dtype=np.float64)
        for i in range(3):
            self._F[i, 4 + i] = 1.0
        # We observe the box, not the velocities.
        self._H = np.zeros((4, 7), dtype=np.float64)
        self._H[:4, :4] = np.eye(4)

        self._P = np.eye(7, dtype=np.float64) * 10.0
        self._P[4:, 4:] *= 1000.0          # velocities start highly uncertain
        self._Q = np.eye(7, dtype=np.float64) * 0.01
        self._Q[4:, 4:] *= 0.01
        self._R = np.eye(4, dtype=np.float64)
        self._R[2:, 2:] *= 10.0            # area/aspect are noisier than the centre

        self._x = np.zeros(7, dtype=np.float64)
        self._x[:4] = self._to_z(box)

    # -- box <-> state ------------------------------------------------------
    @staticmethod
    def _to_z(box: np.ndarray) -> np.ndarray:
        """xyxy -> [cx, cy, area, aspect]."""
        w = max(float(box[2]) - float(box[0]), 1e-6)
        h = max(float(box[3]) - float(box[1]), 1e-6)
        return np.array([float(box[0]) + w / 2.0, float(box[1]) + h / 2.0, w * h, w / h])

    @staticmethod
    def _to_box(z: np.ndarray) -> np.ndarray:
        """[cx, cy, area, aspect] -> xyxy."""
        area = max(float(z[2]), 1e-9)
        aspect = max(float(z[3]), 1e-9)
        w = float(np.sqrt(area * aspect))
        h = area / max(w, 1e-9)
        return np.array([z[0] - w / 2.0, z[1] - h / 2.0,
                         z[0] + w / 2.0, z[1] + h / 2.0], dtype=np.float32)

    # -- filter -------------------------------------------------------------
    def predict(self) -> np.ndarray:
        """Advance one frame and return the predicted box."""
        # A shrinking box can drive the area negative; clamp before it does.
        if self._x[2] + self._x[6] <= 0:
            self._x[6] = 0.0
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self.age += 1
        self.time_since_update += 1
        return self._to_box(self._x[:4])

    def update(self, box: np.ndarray, score: float, label: int, name: str) -> None:
        z = self._to_z(box)
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(7) - K @ self._H) @ self._P

        self.time_since_update = 0
        self.hits += 1
        self.score = float(score)
        self.label = int(label)
        self.name = str(name)

    @property
    def box(self) -> np.ndarray:
        return self._to_box(self._x[:4])


@register("sort")
class SortTracker(BaseTracker):
    """Motion-only baseline tracker. See the module docstring for the algorithm."""

    family = "sort"
    needs_appearance = False

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 1,
                 min_hits: int = 3, variant: Optional[str] = None, **kwargs):
        super().__init__(iou_threshold=iou_threshold, max_age=max_age,
                         min_hits=min_hits, **kwargs)
        if variant not in (None, "", "default"):
            raise ValueError(f"SortTracker has no variant '{variant}'")
        self.reset()

    def reset(self) -> None:
        self._tracks: List[_KalmanBoxTracker] = []
        self._next_id = 1
        self._frame = 0

    # -- association --------------------------------------------------------
    def _associate(self, predicted: np.ndarray, boxes: np.ndarray
                   ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Return (matches, unmatched_detection_indices).

        The Hungarian solution is computed on the full IoU matrix and only then
        filtered by `iou_threshold`. Filtering first would let a greedy leftover win
        an assignment the optimal solution had given to a better pair.
        """
        if len(predicted) == 0 or len(boxes) == 0:
            return [], list(range(len(boxes)))

        from scipy.optimize import linear_sum_assignment

        iou = iou_matrix(predicted, boxes)
        rows, cols = linear_sum_assignment(-iou)
        matches = [(int(r), int(c)) for r, c in zip(rows, cols)
                   if iou[r, c] >= self.iou_threshold]
        matched_dets = {c for _, c in matches}
        return matches, [d for d in range(len(boxes)) if d not in matched_dets]

    def update(self, detection) -> TrackResult:
        self._frame += 1

        boxes = np.asarray(detection.boxes, np.float32).reshape(-1, 4)
        n = len(boxes)
        scores = (np.asarray(detection.scores, np.float32) if len(detection.scores) == n
                  else np.ones(n, np.float32))
        labels = (np.asarray(detection.labels, np.int64) if len(detection.labels) == n
                  else np.zeros(n, np.int64))
        names = (list(detection.names) if len(detection.names) == n
                 else [""] * n)

        predicted = np.array([t.predict() for t in self._tracks], np.float32) \
            if self._tracks else np.zeros((0, 4), np.float32)

        matches, unmatched = self._associate(predicted, boxes)
        for ti, di in matches:
            self._tracks[ti].update(boxes[di], scores[di], labels[di], names[di])
        for di in unmatched:
            self._tracks.append(_KalmanBoxTracker(
                boxes[di], scores[di], labels[di], names[di], self._next_id))
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        # Report only tracks updated this frame. During the first `min_hits` frames a
        # new track is reported immediately, otherwise a sequence would start empty
        # and every object would take an avoidable miss -- this is SORT's own rule.
        live = [t for t in self._tracks
                if t.time_since_update == 0
                and (t.hits >= self.min_hits or self._frame <= self.min_hits)]

        out = TrackResult(
            track_ids=np.array([t.id for t in live], np.int64),
            boxes=(np.array([t.box for t in live], np.float32) if live
                   else np.zeros((0, 4), np.float32)),
            scores=np.array([t.score for t in live], np.float32),
            labels=np.array([t.label for t in live], np.int64),
            names=[t.name for t in live],
        )
        out.check_aligned()
        return out


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
#   python -m DeepDataMiningLearning.ngperception.tracking.trackers.sort
# Two objects crossing plus one appearing late: ids must stay stable and the
# newcomer must get a fresh id.
# ===========================================================================
if __name__ == "__main__":
    from DeepDataMiningLearning.ngdet.detectors.base import Detection

    def det(boxes):
        b = np.array(boxes, np.float32).reshape(-1, 4)
        return Detection(boxes=b, scores=np.ones(len(b), np.float32),
                         labels=np.zeros(len(b), np.int64), names=["car"] * len(b))

    tracker = SortTracker(min_hits=1)
    for t in range(6):
        boxes = [[10 + 12 * t, 20, 60 + 12 * t, 80],
                 [200 - 6 * t, 100, 250 - 6 * t, 160]]
        if t >= 3:
            boxes.append([300, 300 + 5 * (t - 3), 340, 350 + 5 * (t - 3)])
        res = tracker.update(det(boxes))
        print(f"frame {t}: ids={res.track_ids.tolist()}")
