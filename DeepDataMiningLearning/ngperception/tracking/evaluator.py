"""
ngperception.tracking.evaluator
===============================

CLEAR-MOT metrics (Bernardin & Stiefelhagen, 2008) plus IDF1 (Ristani et al., ECCV 2016
workshop) — the standard MOT scoring set, computed per sequence and aggregated the way
MOTChallenge does it: **sum the counts over all frames and sequences, then form the
ratio once**. Averaging per-frame MOTA is a different (and wrong) number, so the
counters are public and the ratios are derived only in `summarize()`.

Definitions, spelled out because the sign conventions differ between papers:

    MOTA = 1 - (FN + FP + IDSW) / GT           [higher better; can be negative]
    MOTP = mean IoU over matched pairs          [higher better, IoU convention]
    IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)      [higher better]
    Recall    = TP / GT
    Precision = TP / (TP + FP)

Matching is per frame, by Hungarian assignment on IoU, with pairs below
`iou_threshold` rejected after the assignment (identical to the association rule in
`trackers/base.py`, so the evaluator does not silently use a stricter or looser gate
than the tracker it scores).

An **ID switch** is counted when a ground-truth object that was previously matched to
tracker id *a* is matched to a different id *b*. Following CLEAR-MOT, the previous
association survives a gap in which the object is unmatched — otherwise every
occlusion would be scored as a switch.

IDF1 is computed globally over the sequence, not per frame: the optimal one-to-one
mapping between ground-truth ids and tracker ids is found by Hungarian assignment on
the number of frames in which each pair co-occurs, and IDTP is the total of the
matched cells. A per-frame approximation would flatter a tracker that swaps ids.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .trackers.base import iou_matrix


def _match(gt_boxes: np.ndarray, tr_boxes: np.ndarray, iou_threshold: float
           ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Hungarian match on IoU, thresholded after assignment. Returns (pairs, iou)."""
    if len(gt_boxes) == 0 or len(tr_boxes) == 0:
        return [], np.zeros((len(gt_boxes), len(tr_boxes)), np.float32)

    from scipy.optimize import linear_sum_assignment

    iou = iou_matrix(gt_boxes, tr_boxes)
    rows, cols = linear_sum_assignment(-iou)
    pairs = [(int(r), int(c)) for r, c in zip(rows, cols)
             if iou[r, c] >= iou_threshold]
    return pairs, iou


class MOTEvaluator:
    """Accumulates CLEAR-MOT counters across frames and sequences.

    Usage
    -----
    ev = MOTEvaluator()
    ev.new_sequence()                     # resets the id-association memory
    for frame in sequence:
        ev.add(gt_ids, gt_boxes, track_ids, track_boxes)
    ev.summarize()
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = float(iou_threshold)
        self.gt = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.idsw = 0
        self._iou_sum = 0.0
        self._last_match: Dict[int, int] = {}
        self._cooccur: Dict[Tuple[int, int], int] = {}
        self._gt_count: Dict[int, int] = {}
        self._tr_count: Dict[int, int] = {}

    def new_sequence(self) -> None:
        """Start a new sequence: ids are only comparable within one."""
        self._last_match = {}

    def add(self, gt_ids: Sequence[int], gt_boxes: np.ndarray,
            track_ids: Sequence[int], track_boxes: np.ndarray) -> None:
        gt_ids = list(gt_ids)
        track_ids = list(track_ids)
        gt_boxes = np.asarray(gt_boxes, np.float32).reshape(-1, 4)
        track_boxes = np.asarray(track_boxes, np.float32).reshape(-1, 4)
        if len(gt_ids) != len(gt_boxes) or len(track_ids) != len(track_boxes):
            raise ValueError("ids and boxes must be index-aligned")

        pairs, iou = _match(gt_boxes, track_boxes, self.iou_threshold)

        self.gt += len(gt_boxes)
        self.tp += len(pairs)
        self.fp += len(track_boxes) - len(pairs)
        self.fn += len(gt_boxes) - len(pairs)

        for g, t in pairs:
            gid, tid = gt_ids[g], track_ids[t]
            self._iou_sum += float(iou[g, t])
            prev = self._last_match.get(gid)
            if prev is not None and prev != tid:
                self.idsw += 1
            self._last_match[gid] = tid
            self._cooccur[(gid, tid)] = self._cooccur.get((gid, tid), 0) + 1

        for gid in gt_ids:
            self._gt_count[gid] = self._gt_count.get(gid, 0) + 1
        for tid in track_ids:
            self._tr_count[tid] = self._tr_count.get(tid, 0) + 1

    # -- IDF1 ---------------------------------------------------------------
    def _idf1(self) -> Tuple[float, int]:
        """Global one-to-one id mapping; returns (idf1, idtp)."""
        if not self._cooccur:
            return 0.0, 0
        from scipy.optimize import linear_sum_assignment

        gids = sorted({g for g, _ in self._cooccur})
        tids = sorted({t for _, t in self._cooccur})
        m = np.zeros((len(gids), len(tids)), np.float64)
        for (g, t), c in self._cooccur.items():
            m[gids.index(g), tids.index(t)] = c

        rows, cols = linear_sum_assignment(-m)
        idtp = int(m[rows, cols].sum())
        idfn = sum(self._gt_count.values()) - idtp
        idfp = sum(self._tr_count.values()) - idtp
        denom = 2 * idtp + idfp + idfn
        return (2.0 * idtp / denom if denom else 0.0), idtp

    def summarize(self, verbose: bool = True) -> Dict[str, float]:
        idf1, idtp = self._idf1()
        out = {
            "MOTA": 1.0 - (self.fn + self.fp + self.idsw) / self.gt if self.gt else 0.0,
            "MOTP": self._iou_sum / self.tp if self.tp else 0.0,
            "IDF1": idf1,
            "recall": self.tp / self.gt if self.gt else 0.0,
            "precision": self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0,
            "GT": float(self.gt), "TP": float(self.tp), "FP": float(self.fp),
            "FN": float(self.fn), "IDSW": float(self.idsw), "IDTP": float(idtp),
        }
        if verbose:
            print(f"  MOTA {out['MOTA']:+.4f}  MOTP {out['MOTP']:.4f}  IDF1 {out['IDF1']:.4f}"
                  f"  |  GT {self.gt}  TP {self.tp}  FP {self.fp}  FN {self.fn}"
                  f"  IDSW {self.idsw}")
        return out


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
#   python -m DeepDataMiningLearning.ngperception.tracking.evaluator
# A perfect tracker must score MOTA 1.0 / IDF1 1.0 / IDSW 0; swapping two ids
# midway must cost exactly two switches and drop IDF1 without touching MOTA's
# detection terms.
# ===========================================================================
if __name__ == "__main__":
    boxes_a = np.array([[0, 0, 10, 10]], np.float32)
    boxes_b = np.array([[50, 50, 60, 60]], np.float32)
    frames = [(np.concatenate([boxes_a, boxes_b]), [1, 2]) for _ in range(6)]

    ev = MOTEvaluator()
    ev.new_sequence()
    for boxes, gids in frames:
        ev.add(gids, boxes, gids, boxes)          # tracker == ground truth
    print("perfect tracker:")
    perfect = ev.summarize()

    ev2 = MOTEvaluator()
    ev2.new_sequence()
    for i, (boxes, gids) in enumerate(frames):
        tids = gids if i < 3 else [2, 1]           # ids swapped from frame 3
        ev2.add(gids, boxes, tids, boxes)
    print("ids swapped at frame 3:")
    swapped = ev2.summarize()

    assert perfect["MOTA"] == 1.0 and perfect["IDF1"] == 1.0 and perfect["IDSW"] == 0
    assert swapped["IDSW"] == 2, swapped["IDSW"]
    assert swapped["FP"] == 0 and swapped["FN"] == 0
    assert swapped["IDF1"] < perfect["IDF1"]
    print("OK")
