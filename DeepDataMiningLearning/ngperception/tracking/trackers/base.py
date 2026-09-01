"""
ngperception.tracking.trackers.base
===================================

The pluggable multi-object-tracker contract — the tracking analogue of
`ngdet.detectors.base` and `ngperception.depth.estimators.base`. Every backend (SORT,
DeepSORT, ByteTrack, OC-SORT, ...) is wrapped in a small adapter that subclasses
`BaseTracker` and registers itself with `@register("name")`.

A tracker's ONLY job is:
    per-frame `ngdet.Detection`  ->  `TrackResult`  (the same boxes, now carrying ids)

It does **not** detect. It consumes whatever `ngdet` produced for the frame, so the same
sequence can be re-tracked under a different detector without re-running the tracker's
own logic — which is what makes the detector/tracker ablation in `run_eval.py` fair.

Two families exist, and the difference matters for what a backend may import:

* **motion-only** trackers (SORT, ByteTrack, OC-SORT) associate on geometry alone —
  IoU plus a motion model. They are pure numpy/scipy: no network, no weights, no GPU.
* **appearance** trackers (DeepSORT, BoT-SORT) add a re-identification embedding and
  therefore need a model. Their heavy imports belong INSIDE the subclass `__init__`,
  never at module top level, so that `import ngperception.tracking` stays cheap and a
  missing optional dependency only breaks the backend that needs it.

Identity semantics every adapter must honour:

* an id is issued once and never reused within a sequence;
* the same physical object keeps its id across frames while it is tracked;
* a track that has not been matched for longer than the backend's `max_age` is retired,
  and if the object reappears it is a *new* id — recovering the old one is re-identification,
  which is a different (appearance) family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type

import numpy as np


@dataclass
class TrackResult:
    """Confirmed tracks for ONE frame.

    All fields are aligned by index (the i-th box has the i-th id/score/label/name),
    the same contract `ngdet.Detection` uses. Boxes are absolute pixel coordinates in
    xyxy (x_min, y_min, x_max, y_max) in the original image frame, so a `TrackResult`
    can be overlaid on the source image exactly like a `Detection`.

    Attributes
    ----------
    track_ids : np.ndarray
        int64, one id per row of `boxes`. Unique within a sequence.
    boxes : np.ndarray
        Nx4 float32, xyxy. For a matched track this is the filtered (smoothed) estimate,
        not the raw detection — that is the point of the motion model.
    scores : np.ndarray
        float32, carried through from the detection that updated the track. A track
        coasting on prediction alone is not reported, so every row has a real score.
    labels : np.ndarray
        int64 unified-taxonomy ids, carried through from the detection.
    names : list of str
        Human-readable class names, index-aligned with `labels`.
    """

    track_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    names: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.boxes)

    def check_aligned(self) -> None:
        """Raise if the index-alignment contract is broken.

        Cheap enough to call in tests and in a backend's own `__main__` block; an
        adapter that filters `boxes` but forgets `labels` is exactly the bug this
        catches, and it is silent otherwise.
        """
        n = len(self.boxes)
        for name, arr in (("track_ids", self.track_ids), ("scores", self.scores),
                          ("labels", self.labels), ("names", self.names)):
            if len(arr) != n:
                raise ValueError(
                    f"TrackResult is misaligned: {n} boxes but {len(arr)} {name}")
        if len(set(self.track_ids.tolist())) != n:
            raise ValueError("TrackResult contains duplicate track ids in one frame")


class BaseTracker:
    """Abstract base for all tracker adapters.

    Subclasses implement `update`. A tracker is stateful across a sequence, so
    `reset()` must return it to the state it had at construction — `run_eval.py` reuses
    one instance across sequences and relies on that.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU for a detection to be accepted as the continuation of a track.
    max_age : int
        Frames a track may go unmatched before it is retired.
    min_hits : int
        Matches required before a track is reported. Suppresses one-frame false
        positives at the cost of a short delay on genuinely new objects.
    """

    #: human-readable backend family name (set by subclass)
    family: str = "base"
    #: True for trackers that need an appearance/re-id model (and therefore a device)
    needs_appearance: bool = False

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 1,
                 min_hits: int = 3, **kwargs):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)

    def update(self, detection) -> TrackResult:
        """Advance the tracker by one frame and return the confirmed tracks.

        `detection` is an `ngdet.detectors.base.Detection`. Must be called once per
        frame, in order; skipping frames breaks the motion model.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Forget all tracks and restart id numbering for a new sequence."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Geometry shared by every motion-only backend.
# ---------------------------------------------------------------------------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes -> (len(a), len(b)) float32.

    Degenerate boxes (zero width or height) give 0 rather than a division warning,
    and an empty input gives a correctly-shaped empty matrix so callers do not have
    to special-case it.
    """
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Registry: short key -> adapter class, so a CLI can spell "--trackers sort" and
# resolve the class by the part before the colon (identical convention to
# ngdet.detectors and ngperception.depth.estimators).
# ---------------------------------------------------------------------------
TRACKER_REGISTRY: Dict[str, Type[BaseTracker]] = {}


def register(name: str) -> Callable[[Type[BaseTracker]], Type[BaseTracker]]:
    def deco(cls: Type[BaseTracker]) -> Type[BaseTracker]:
        TRACKER_REGISTRY[name] = cls
        return cls
    return deco


def build_tracker(spec: str, **kwargs) -> BaseTracker:
    """Instantiate a tracker from a "key" or "key:variant" spec string.

    Examples
    --------
    build_tracker("sort")
    build_tracker("sort", iou_threshold=0.2, max_age=5)
    """
    from . import sort  # noqa: F401  (side effect: register backends)

    key, variant = (spec.split(":", 1) + [None])[:2] if ":" in spec else (spec, None)
    if key not in TRACKER_REGISTRY:
        raise KeyError(
            f"Unknown tracker backend '{key}'. Registered: {list(TRACKER_REGISTRY)}")
    if variant is not None:
        kwargs.setdefault("variant", variant)
    return TRACKER_REGISTRY[key](**kwargs)


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
#   python -m DeepDataMiningLearning.ngperception.tracking.trackers.base
# Expected: prints the registered tracker backends and an IoU sanity check.
# ===========================================================================
if __name__ == "__main__":
    # Under `python -m`, this file is loaded as `__main__`; importing an adapter
    # loads it a *second* time under its real name, and `@register` populates that
    # copy's registry, not this one. Read the canonical module so the self-test
    # reports what a normal `import` would see rather than an empty list.
    from DeepDataMiningLearning.ngperception.tracking.trackers import (  # noqa: F401
        base as _canonical, sort,
    )
    print("Registered tracker backends:", list(_canonical.TRACKER_REGISTRY))

    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10],       # identical      -> 1.0
                  [5, 0, 15, 10],       # half overlap   -> 1/3
                  [20, 20, 30, 30],     # disjoint       -> 0.0
                  [0, 0, 10, 0]], np.float32)   # degenerate -> 0.0
    print("IoU row:", np.round(iou_matrix(a, b)[0], 4).tolist(),
          "(expected [1.0, 0.3333, 0.0, 0.0])")
