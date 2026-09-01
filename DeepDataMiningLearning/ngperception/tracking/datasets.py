"""
ngperception.tracking.datasets
==============================

Tracking sequences in a model-agnostic form: a sequence yields, per frame, the
detections a tracker consumes and the ground truth an evaluator scores against.

The on-disk format is **MOTChallenge**, which every MOT benchmark (MOT16/17/20,
DanceTrack, KITTI-MOT via its converter) either uses or exports to. One CSV row per
object per frame:

    frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z

Two conversions happen here and both are places bugs hide, so they are done once, in
one place, and tested:

* **frames are 1-indexed** in the file and 0-indexed in this API;
* **boxes are xywh** (left, top, width, height) in the file and **xyxy** everywhere in
  `ngdet`/`ngperception`, so ``x2 = left + width`` and ``y2 = top + height`` —
  *not* ``left + width - 1``. The MOTChallenge devkit treats the box as a continuous
  rectangle, not an inclusive pixel range, and an off-by-one here shifts every IoU.

`gt.txt` additionally uses `conf` as a **flag**: 0 marks an ignored/distractor box that
must not be scored. Those rows are dropped by default (`keep_ignored=False`), because
counting them would turn correct behaviour into false positives.

No dataset ships with this repository. `synthetic_sequence()` produces a sequence with
the same structure so the module, the tracker and the evaluator can all be exercised
without downloading anything — that is what the self-test below and the unit tests use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

DEFAULT_MOT_ROOT = "/mnt/e/Shared/Dataset/MOT17/"


@dataclass
class TrackingFrame:
    """One frame of a sequence. All arrays are index-aligned.

    `det_boxes` is what the tracker sees; `gt_boxes`/`gt_ids` are what the evaluator
    scores against. A frame may have either side empty.
    """

    frame_id: int
    det_boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    det_scores: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    gt_boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    gt_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """MOTChallenge (left, top, w, h) -> (x1, y1, x2, y2). Width is a length, not a span."""
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    out = np.empty_like(b)
    out[:, 0] = b[:, 0]
    out[:, 1] = b[:, 1]
    out[:, 2] = b[:, 0] + b[:, 2]
    out[:, 3] = b[:, 1] + b[:, 3]
    return out


def read_mot_csv(path: str, keep_ignored: bool = False
                 ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read a MOTChallenge csv -> {frame_index: (boxes_xyxy, ids, confs)}.

    Frame indices in the returned dict are **0-based**. Rows whose `conf` is 0 are
    dropped unless `keep_ignored` is set.
    """
    per_frame: Dict[int, List[Tuple[float, ...]]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split(",")
            if len(parts) < 7:
                raise ValueError(f"{path}: expected >=7 columns, got {len(parts)}: {raw!r}")
            frame = int(float(parts[0])) - 1          # file is 1-indexed
            oid = int(float(parts[1]))
            l, t, w, h = (float(x) for x in parts[2:6])
            conf = float(parts[6])
            if not keep_ignored and conf == 0.0:
                continue
            per_frame.setdefault(frame, []).append((l, t, w, h, conf, oid))

    out: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for frame, rows in per_frame.items():
        arr = np.array([r[:4] for r in rows], np.float32)
        out[frame] = (xywh_to_xyxy(arr),
                      np.array([r[5] for r in rows], np.int64),
                      np.array([r[4] for r in rows], np.float32))
    return out


class MOTSequence:
    """One MOTChallenge sequence directory: `<root>/<name>/{gt/gt.txt,det/det.txt}`.

    Iterating yields `TrackingFrame` in frame order, including frames where one side
    is empty — a tracker must see those to age its tracks correctly.
    """

    def __init__(self, root: str = DEFAULT_MOT_ROOT, name: str = "MOT17-02",
                 keep_ignored: bool = False):
        self.root, self.name = root, name
        seq_dir = os.path.join(root, name)
        gt_path = os.path.join(seq_dir, "gt", "gt.txt")
        det_path = os.path.join(seq_dir, "det", "det.txt")
        if not os.path.isdir(seq_dir):
            raise FileNotFoundError(
                f"Sequence directory not found: {seq_dir}. Point --root at a "
                "MOTChallenge-style dataset, or use synthetic_sequence() for a "
                "dataset-free run.")
        self._gt = read_mot_csv(gt_path, keep_ignored) if os.path.exists(gt_path) else {}
        self._det = read_mot_csv(det_path, keep_ignored=True) if os.path.exists(det_path) else {}
        frames = set(self._gt) | set(self._det)
        self._frames = sorted(frames)

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[TrackingFrame]:
        for f in self._frames:
            gb, gi, _ = self._gt.get(f, (np.zeros((0, 4), np.float32),
                                         np.zeros((0,), np.int64),
                                         np.zeros((0,), np.float32)))
            db, _, dc = self._det.get(f, (np.zeros((0, 4), np.float32),
                                          np.zeros((0,), np.int64),
                                          np.zeros((0,), np.float32)))
            yield TrackingFrame(frame_id=f, det_boxes=db, det_scores=dc,
                                gt_boxes=gb, gt_ids=gi)


def synthetic_sequence(n_frames: int = 20, n_objects: int = 3, miss_rate: float = 0.0,
                       jitter: float = 0.0, seed: int = 0) -> List[TrackingFrame]:
    """A dataset-free sequence with known ground truth.

    Objects move linearly. `miss_rate` drops detections (never ground truth), and
    `jitter` perturbs detection corners in pixels — so a caller can dial in exactly
    the failure the tracker is supposed to survive.
    """
    rng = np.random.default_rng(seed)
    starts = np.array([[40 + 130 * i, 40 + 60 * i, 100 + 130 * i, 120 + 60 * i]
                       for i in range(n_objects)], np.float32)
    vels = np.array([[7 - 3 * i, 2 + i, 7 - 3 * i, 2 + i] for i in range(n_objects)],
                    np.float32)

    frames: List[TrackingFrame] = []
    for f in range(n_frames):
        gt = starts + vels * f
        keep = rng.random(n_objects) >= miss_rate
        det = gt[keep].copy()
        if jitter and len(det):
            det += rng.uniform(-jitter, jitter, det.shape).astype(np.float32)
        frames.append(TrackingFrame(
            frame_id=f,
            det_boxes=det.astype(np.float32),
            det_scores=np.ones(len(det), np.float32),
            gt_boxes=gt.astype(np.float32),
            gt_ids=np.arange(1, n_objects + 1, dtype=np.int64),
        ))
    return frames


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
#   python -m DeepDataMiningLearning.ngperception.tracking.datasets
# Round-trips a MOTChallenge csv through the reader and prints a synthetic sequence.
# ===========================================================================
if __name__ == "__main__":
    import tempfile

    csv = ("1,1,10,20,30,40,1,-1,-1,-1\n"
           "1,2,100,100,50,50,1,-1,-1,-1\n"
           "2,1,12,22,30,40,1,-1,-1,-1\n"
           "2,9,500,500,10,10,0,-1,-1,-1\n")     # conf=0 -> ignored
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "gt.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(csv)
        parsed = read_mot_csv(p)

    print("frames parsed:", sorted(parsed))                 # 0-based -> [0, 1]
    boxes, ids, _ = parsed[0]
    print("frame 0 ids:", ids.tolist(), "boxes:", boxes.tolist())
    assert sorted(parsed) == [0, 1], "frames must be 0-based"
    assert boxes[0].tolist() == [10.0, 20.0, 40.0, 60.0], boxes[0].tolist()
    assert 9 not in parsed[1][1].tolist(), "conf=0 row must be dropped"

    seq = synthetic_sequence(n_frames=4, n_objects=2)
    for fr in seq:
        print(f"  frame {fr.frame_id}: {len(fr.det_boxes)} det, {len(fr.gt_boxes)} gt")
    print("OK")
