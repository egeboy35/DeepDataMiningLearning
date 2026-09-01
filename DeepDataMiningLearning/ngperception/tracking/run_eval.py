"""
ngperception.tracking.run_eval
==============================

Compare tracker backends on the same sequences under identical settings — the
tracking analogue of `ngperception/depth/run_eval.py`.

Every backend sees the **same detections**, so a difference in MOTA/IDF1 is a
difference in association, not in detection quality. That is the only comparison the
suite claims to make; swapping in a stronger detector moves every row at once.

Examples
--------
    # dataset-free smoke run: synthetic sequences, no download, seconds
    python -m DeepDataMiningLearning.ngperception.tracking.run_eval \\
        --trackers sort --synthetic

    # sweep the association gate on synthetic data with detector noise
    python -m DeepDataMiningLearning.ngperception.tracking.run_eval \\
        --trackers sort --synthetic --jitter 6 --miss-rate 0.1

    # MOTChallenge-style directory
    python -m DeepDataMiningLearning.ngperception.tracking.run_eval \\
        --trackers sort --root /data/MOT17 --sequences MOT17-02 MOT17-04
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

from DeepDataMiningLearning.ngdet.detectors.base import Detection

from .datasets import DEFAULT_MOT_ROOT, MOTSequence, TrackingFrame, synthetic_sequence
from .evaluator import MOTEvaluator
from .trackers.base import build_tracker

DEFAULT_OUT_DIR = "DeepDataMiningLearning/ngperception/output/tracking"


def _as_detection(frame: TrackingFrame) -> Detection:
    n = len(frame.det_boxes)
    return Detection(
        boxes=frame.det_boxes,
        scores=(frame.det_scores if len(frame.det_scores) == n
                else np.ones(n, np.float32)),
        labels=np.zeros(n, np.int64),
        names=["object"] * n,
    )


def evaluate(tracker_spec: str, sequences: Dict[str, List[TrackingFrame]],
             eval_iou: float, **tracker_kwargs) -> Dict[str, float]:
    """Run one backend over every sequence and return the aggregated metrics.

    Counters accumulate across sequences and the ratios are formed once at the end,
    which is the MOTChallenge convention -- averaging per-sequence MOTA weights a
    10-frame clip the same as a 1000-frame one.

    `eval_iou` is the *scoring* gate; the tracker's own association gate arrives in
    `tracker_kwargs` as `iou_threshold`. They are deliberately different names: the
    two thresholds mean different things and calling both `iou_threshold` collides.
    """
    tracker = build_tracker(tracker_spec, **tracker_kwargs)
    ev = MOTEvaluator(iou_threshold=eval_iou)

    for _, frames in sequences.items():
        tracker.reset()
        ev.new_sequence()
        for fr in frames:
            res = tracker.update(_as_detection(fr))
            ev.add(fr.gt_ids.tolist(), fr.gt_boxes,
                   res.track_ids.tolist(), res.boxes)
    return ev.summarize(verbose=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="ngperception tracking comparison.")
    ap.add_argument("--trackers", nargs="+", default=["sort"],
                    help='backend specs, e.g. "sort" (see TRACKER_REGISTRY)')
    ap.add_argument("--root", default=DEFAULT_MOT_ROOT)
    ap.add_argument("--sequences", nargs="*", default=None,
                    help="sequence directory names under --root")
    ap.add_argument("--synthetic", action="store_true",
                    help="ignore --root and score on generated sequences (no dataset)")
    ap.add_argument("--synthetic-frames", type=int, default=30)
    ap.add_argument("--synthetic-objects", type=int, default=4)
    ap.add_argument("--miss-rate", type=float, default=0.0,
                    help="synthetic only: fraction of detections dropped per frame")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="synthetic only: uniform pixel noise on detection corners")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iou-threshold", type=float, default=0.5,
                    help="evaluation gate (MOTChallenge uses 0.5)")
    ap.add_argument("--track-iou", type=float, default=0.3,
                    help="association gate inside the tracker")
    ap.add_argument("--max-age", type=int, default=1)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    if args.synthetic:
        sequences = {
            f"synthetic-{i}": synthetic_sequence(
                n_frames=args.synthetic_frames, n_objects=args.synthetic_objects,
                miss_rate=args.miss_rate, jitter=args.jitter, seed=args.seed + i)
            for i in range(2)
        }
        source = (f"synthetic ({args.synthetic_frames} frames x "
                  f"{args.synthetic_objects} objects, miss={args.miss_rate}, "
                  f"jitter={args.jitter}, seed={args.seed})")
    else:
        names = args.sequences or []
        if not names:
            ap.error("give --sequences NAME [NAME ...] or use --synthetic")
        sequences = {n: list(MOTSequence(args.root, n)) for n in names}
        source = f"{args.root} ({', '.join(names)})"

    n_frames = sum(len(v) for v in sequences.values())
    print(f"source : {source}")
    print(f"frames : {n_frames} across {len(sequences)} sequence(s)")
    print(f"eval   : IoU>={args.iou_threshold}   tracker: IoU>={args.track_iou} "
          f"max_age={args.max_age} min_hits={args.min_hits}")
    print()
    print(f"  {'tracker':<12s} {'MOTA':>9s} {'MOTP':>7s} {'IDF1':>7s} "
          f"{'IDSW':>6s} {'FP':>7s} {'FN':>7s}")

    results: Dict[str, Dict[str, float]] = {}
    for spec in args.trackers:
        m = evaluate(spec, sequences, args.iou_threshold,
                     iou_threshold=args.track_iou, max_age=args.max_age,
                     min_hits=args.min_hits)
        results[spec] = m
        print(f"  {spec:<12s} {m['MOTA']:>+9.4f} {m['MOTP']:>7.4f} {m['IDF1']:>7.4f} "
              f"{int(m['IDSW']):>6d} {int(m['FP']):>7d} {int(m['FN']):>7d}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "tracking_metrics.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"source": source, "settings": vars(args), "results": results},
                  fh, indent=2)
    print(f"\nwrote {out_path}")


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
#   python -m DeepDataMiningLearning.ngperception.tracking.run_eval --trackers sort --synthetic
# Expected: a clean synthetic sequence scores MOTA close to 1.0 with 0 ID switches.
# ===========================================================================
if __name__ == "__main__":
    main()
