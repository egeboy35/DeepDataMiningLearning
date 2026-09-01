# ngperception/tracking — multi-object tracking on top of `ngdet` detections

The **basic** tier of the tracking task: SORT, the motion-only baseline every later
method is measured against. It runs on **numpy + scipy alone** — no network, no weights,
no GPU, no dataset download — so the whole arm is reproducible by anyone in seconds,
and a later appearance-based backend has a like-for-like reference to beat.

The module mirrors the three-layer shape the rest of the suite uses
(**trackers / datasets / evaluator**), and consumes `ngdet.Detection` rather than
detecting anything itself — so the same sequence can be re-tracked under a different
detector without touching the tracker, which is what makes the ablation fair.

## Layout

```
tracking/
├── trackers/base.py    # TrackResult + BaseTracker + TRACKER_REGISTRY + build_tracker + iou_matrix
├── trackers/sort.py    # @register("sort") — Kalman (cx,cy,area,aspect) + Hungarian on IoU
├── datasets.py         # MOTChallenge csv reader (+ a dataset-free synthetic generator)
├── evaluator.py        # CLEAR-MOT: MOTA / MOTP / IDF1 / ID-switches, MOTChallenge aggregation
├── run_eval.py         # CLI, same shape as depth/run_eval.py
└── tests/              # 26 tests, CPU only, ~1.5 s
```

## Quick start — no dataset needed

```bash
python -m DeepDataMiningLearning.ngperception.tracking.run_eval \
    --trackers sort --synthetic --min-hits 1
```

```
  tracker           MOTA    MOTP    IDF1   IDSW      FP      FN
  sort           +1.0000  1.0000  1.0000      0       0       0
```

Dial in the failure you want the tracker to survive:

| condition | MOTA | MOTP | IDF1 | IDSW | FN |
|---|---:|---:|---:|---:|---:|
| clean | +1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 6 px detector jitter | +1.0000 | 0.9187 | 1.0000 | 0 | 0 |
| 10 % missed detections | +0.9000 | 1.0000 | 0.9474 | 0 | 24 |
| 15 % missed + 8 px jitter | +0.8583 | 0.8914 | 0.7175 | 0 | 34 |

*(`--min-hits 1 --max-age 3`, 2 × 30 frames × 4 objects.)* Jitter degrades **MOTP** —
localisation — while leaving MOTA's detection terms alone; missed detections cost
**MOTA** in proportion. That separation is the point of reporting both.

## On a MOTChallenge-style dataset

```bash
python -m DeepDataMiningLearning.ngperception.tracking.run_eval \
    --trackers sort --root /data/MOT17 --sequences MOT17-02 MOT17-04
```

`datasets.py` reads `<root>/<seq>/{gt/gt.txt,det/det.txt}` and does the two conversions
that format needs, in one place, tested: frames are **1-indexed on disk, 0-indexed in the
API**, and boxes are **xywh on disk, xyxy everywhere in `ngdet`/`ngperception`**. Rows
whose `conf` is `0` are MOTChallenge's ignore flag and are dropped by default.

## Algorithm

1. a **constant-velocity Kalman filter** per track over `[cx, cy, area, aspect]`, with
   aspect held constant — SORT's own simplification, kept so the baseline is the
   published one and not a private variant;
2. **Hungarian assignment** on the IoU between each track's *predicted* box and each
   detection. Matches below `--track-iou` are rejected **after** the assignment, not
   before: filtering first would let a leftover pair win an assignment the optimal
   solution had given to a better one.

## What this baseline does *not* do

Stated so the numbers are not over-read:

- **no re-identification** — an object that leaves and returns gets a new id, and the
  sequence takes an ID-switch. Recovering the old id is the appearance family's job
  (DeepSORT, BoT-SORT);
- **no occlusion reasoning** beyond coasting for `--max-age` frames;
- it inherits every miss of the detector it is given — which is why `run_eval.py` feeds
  every backend the *same* detections.

## Adding a backend

```python
from .base import BaseTracker, TrackResult, register

@register("bytetrack")
class ByteTrack(BaseTracker):
    family = "bytetrack"
    def update(self, detection) -> TrackResult: ...
    def reset(self) -> None: ...
```

Heavy imports (torch, a re-id model) belong **inside** the subclass `__init__`, never at
module top level, so `import ngperception.tracking` stays cheap and a missing optional
dependency only breaks the backend that needs it — the same rule `ngdet.detectors` uses.

## Tests

```bash
python -m pytest DeepDataMiningLearning/ngperception/tracking/tests -q
```

26 tests, ~1.5 s, no GPU and no dataset. They cover the IoU edge cases (identical, half,
disjoint, degenerate, empty), id stability and id reuse, survival across a gap shorter
than `max_age` and retirement past it, the index-alignment contract, the MOTChallenge
conversions, and the CLEAR-MOT counters — including that a gap is **not** scored as an
ID switch and that an id swap costs exactly two.
