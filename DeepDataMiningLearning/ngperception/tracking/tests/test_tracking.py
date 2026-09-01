"""Tests for ngperception.tracking — CPU only, no model, no dataset, seconds to run."""

from __future__ import annotations

import numpy as np
import pytest

from DeepDataMiningLearning.ngdet.detectors.base import Detection
from DeepDataMiningLearning.ngperception.tracking.evaluator import MOTEvaluator
from DeepDataMiningLearning.ngperception.tracking.trackers import sort as _sort  # noqa: F401
from DeepDataMiningLearning.ngperception.tracking.trackers.base import (
    TRACKER_REGISTRY, TrackResult, build_tracker, iou_matrix,
)


def det(boxes, scores=None, labels=None, names=None):
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    n = len(b)
    return Detection(
        boxes=b,
        scores=np.asarray(scores if scores is not None else [1.0] * n, np.float32),
        labels=np.asarray(labels if labels is not None else [0] * n, np.int64),
        names=list(names) if names is not None else ["car"] * n,
    )


# ------------------------------------------------------------------ geometry
def test_iou_identical_half_and_disjoint():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    got = iou_matrix(a, b)[0]
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(1 / 3, rel=1e-4)   # 50 inter / 150 union
    assert got[2] == pytest.approx(0.0)


def test_iou_degenerate_box_is_zero_not_nan():
    a = np.array([[0, 0, 10, 0]], np.float32)          # zero height
    b = np.array([[0, 0, 10, 10]], np.float32)
    got = iou_matrix(a, b)
    assert np.isfinite(got).all() and got[0, 0] == 0.0


def test_iou_empty_inputs_keep_their_shape():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)
    assert iou_matrix(np.zeros((2, 4)), np.zeros((0, 4))).shape == (2, 0)


# ------------------------------------------------------------------ registry
def test_sort_is_registered_and_buildable():
    assert "sort" in TRACKER_REGISTRY
    t = build_tracker("sort", iou_threshold=0.25, max_age=5)
    assert t.iou_threshold == 0.25 and t.max_age == 5


def test_unknown_backend_names_the_registered_ones():
    with pytest.raises(KeyError, match="sort"):
        build_tracker("deepsort")


# ------------------------------------------------------------------ identity
def test_ids_are_stable_while_objects_move():
    t = build_tracker("sort", min_hits=1)
    ids_per_frame = []
    for f in range(8):
        ids_per_frame.append(t.update(det([
            [10 + 12 * f, 20, 60 + 12 * f, 80],
            [200 - 6 * f, 100, 250 - 6 * f, 160],
        ])).track_ids.tolist())
    assert all(len(ids) == 2 for ids in ids_per_frame)
    assert len({tuple(ids) for ids in ids_per_frame}) == 1, ids_per_frame


def test_new_object_gets_a_fresh_id_and_keeps_it():
    t = build_tracker("sort", min_hits=1)
    seen = []
    for f in range(6):
        boxes = [[10 + 12 * f, 20, 60 + 12 * f, 80]]
        if f >= 3:
            boxes.append([300, 300 + 5 * (f - 3), 340, 350 + 5 * (f - 3)])
        seen.append(t.update(det(boxes)).track_ids.tolist())
    assert len(seen[2]) == 1 and len(seen[3]) == 2
    newcomer = seen[3][1]
    assert newcomer not in seen[2]
    assert all(newcomer in ids for ids in seen[3:])


def test_track_survives_a_gap_shorter_than_max_age():
    """A miss inside `max_age` must not end the track, and the id must survive it.

    Without this the suite passes even when `max_age` is ignored entirely -- a
    mutation that retires every unmatched track immediately went undetected until
    this case was added.
    """
    t = build_tracker("sort", min_hits=1, max_age=3)
    first = t.update(det([[0, 0, 20, 20]])).track_ids.tolist()
    assert first == [1]
    for _ in range(2):                                   # two missed frames < max_age
        assert len(t.update(det(np.zeros((0, 4), np.float32)))) == 0
    again = t.update(det([[2, 2, 22, 22]])).track_ids.tolist()
    assert again == first, f"track should have survived the gap, got {again}"


def test_track_dies_after_a_gap_longer_than_max_age():
    t = build_tracker("sort", min_hits=1, max_age=2)
    first = t.update(det([[0, 0, 20, 20]])).track_ids.tolist()
    for _ in range(4):                                   # longer than max_age
        t.update(det(np.zeros((0, 4), np.float32)))
    again = t.update(det([[0, 0, 20, 20]])).track_ids.tolist()
    assert again != first, "track should have been retired past max_age"


def test_ids_are_never_reused_after_a_track_dies():
    t = build_tracker("sort", min_hits=1, max_age=1)
    first = t.update(det([[0, 0, 20, 20]])).track_ids.tolist()
    for _ in range(4):                                   # object disappears
        t.update(det(np.zeros((0, 4), np.float32)))
    again = t.update(det([[0, 0, 20, 20]])).track_ids.tolist()
    assert again and again[0] not in first


def test_reset_restarts_the_sequence():
    t = build_tracker("sort", min_hits=1)
    a = t.update(det([[0, 0, 20, 20]])).track_ids.tolist()
    t.reset()
    b = t.update(det([[0, 0, 20, 20]])).track_ids.tolist()
    assert a == b == [1]


# ------------------------------------------------------------------ contract
def test_track_result_stays_index_aligned():
    t = build_tracker("sort", min_hits=1)
    res = t.update(det([[0, 0, 10, 10], [50, 50, 70, 70]],
                       scores=[0.9, 0.4], labels=[3, 7], names=["car", "bus"]))
    res.check_aligned()
    assert len(res) == 2
    assert set(res.labels.tolist()) == {3, 7}
    assert set(res.names) == {"car", "bus"}


def test_check_aligned_catches_a_broken_result():
    bad = TrackResult(track_ids=np.array([1, 2]),
                      boxes=np.zeros((2, 4), np.float32),
                      scores=np.array([1.0], np.float32),      # one short
                      labels=np.zeros(2, np.int64), names=["a", "b"])
    with pytest.raises(ValueError, match="misaligned"):
        bad.check_aligned()


def test_empty_detection_yields_empty_result():
    t = build_tracker("sort", min_hits=1)
    res = t.update(det(np.zeros((0, 4), np.float32)))
    assert len(res) == 0
    res.check_aligned()


# ------------------------------------------------------------------ evaluator
def test_perfect_tracker_scores_one():
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], np.float32)
    ev = MOTEvaluator()
    ev.new_sequence()
    for _ in range(6):
        ev.add([1, 2], boxes, [1, 2], boxes)
    s = ev.summarize(verbose=False)
    assert s["MOTA"] == pytest.approx(1.0)
    assert s["IDF1"] == pytest.approx(1.0)
    assert s["IDSW"] == 0 and s["FP"] == 0 and s["FN"] == 0


def test_id_swap_costs_exactly_two_switches():
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], np.float32)
    ev = MOTEvaluator()
    ev.new_sequence()
    for i in range(6):
        ev.add([1, 2], boxes, [1, 2] if i < 3 else [2, 1], boxes)
    s = ev.summarize(verbose=False)
    assert s["IDSW"] == 2
    assert s["FP"] == 0 and s["FN"] == 0          # detection is untouched
    assert s["MOTA"] == pytest.approx(1 - 2 / 12)
    assert s["IDF1"] < 1.0


def test_a_gap_is_not_scored_as_an_id_switch():
    """CLEAR-MOT keeps the association across frames where the object is unmatched."""
    box = np.array([[0, 0, 10, 10]], np.float32)
    ev = MOTEvaluator()
    ev.new_sequence()
    ev.add([1], box, [7], box)
    ev.add([1], box, [], np.zeros((0, 4), np.float32))   # missed this frame
    ev.add([1], box, [7], box)                            # same id returns
    s = ev.summarize(verbose=False)
    assert s["IDSW"] == 0
    assert s["FN"] == 1


def test_false_positive_and_miss_are_counted_separately():
    gt = np.array([[0, 0, 10, 10]], np.float32)
    tr = np.array([[900, 900, 910, 910]], np.float32)     # nowhere near
    ev = MOTEvaluator()
    ev.new_sequence()
    ev.add([1], gt, [1], tr)
    s = ev.summarize(verbose=False)
    assert s["FP"] == 1 and s["FN"] == 1 and s["TP"] == 0
    assert s["MOTA"] == pytest.approx(-1.0)               # MOTA may go negative


# ------------------------------------------------------------------ end to end
def test_sort_tracks_a_clean_sequence_at_mota_one():
    """Feed the tracker its own ground truth; it must reproduce it exactly."""
    t = build_tracker("sort", min_hits=1)
    ev = MOTEvaluator()
    ev.new_sequence()
    for f in range(10):
        boxes = np.array([[10 + 8 * f, 20, 60 + 8 * f, 80],
                          [400 - 7 * f, 200, 460 - 7 * f, 260]], np.float32)
        res = t.update(det(boxes))
        ev.add([1, 2], boxes, res.track_ids.tolist(), res.boxes)
    s = ev.summarize(verbose=False)
    assert s["IDSW"] == 0
    assert s["MOTA"] > 0.99, s
    assert s["IDF1"] > 0.99, s


# ------------------------------------------------------------------ datasets
def test_xywh_to_xyxy_treats_width_as_a_length():
    from DeepDataMiningLearning.ngperception.tracking.datasets import xywh_to_xyxy
    got = xywh_to_xyxy([[10, 20, 30, 40]])[0].tolist()
    assert got == [10.0, 20.0, 40.0, 60.0]      # not 39/59: w is a length, not a span


def test_mot_csv_is_one_indexed_on_disk_and_zero_indexed_in_the_api(tmp_path):
    from DeepDataMiningLearning.ngperception.tracking.datasets import read_mot_csv
    p = tmp_path / "gt.txt"
    p.write_text("1,1,10,20,30,40,1,-1,-1,-1\n"
                 "2,1,12,22,30,40,1,-1,-1,-1\n", encoding="utf-8")
    parsed = read_mot_csv(str(p))
    assert sorted(parsed) == [0, 1]


def test_mot_csv_drops_ignored_rows_unless_asked(tmp_path):
    from DeepDataMiningLearning.ngperception.tracking.datasets import read_mot_csv
    p = tmp_path / "gt.txt"
    p.write_text("1,1,10,20,30,40,1,-1,-1,-1\n"
                 "1,9,500,500,10,10,0,-1,-1,-1\n", encoding="utf-8")   # conf=0
    assert read_mot_csv(str(p))[0][1].tolist() == [1]
    assert sorted(read_mot_csv(str(p), keep_ignored=True)[0][1].tolist()) == [1, 9]


def test_mot_csv_rejects_a_short_row(tmp_path):
    from DeepDataMiningLearning.ngperception.tracking.datasets import read_mot_csv
    p = tmp_path / "gt.txt"
    p.write_text("1,1,10,20\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        read_mot_csv(str(p))


def test_missing_sequence_directory_says_what_to_do(tmp_path):
    from DeepDataMiningLearning.ngperception.tracking.datasets import MOTSequence
    with pytest.raises(FileNotFoundError, match="synthetic_sequence"):
        MOTSequence(root=str(tmp_path), name="NOPE-01")


def test_synthetic_sequence_is_reproducible_and_shaped():
    from DeepDataMiningLearning.ngperception.tracking.datasets import synthetic_sequence
    a = synthetic_sequence(n_frames=5, n_objects=3, jitter=4, seed=7)
    b = synthetic_sequence(n_frames=5, n_objects=3, jitter=4, seed=7)
    assert len(a) == 5
    for fa, fb in zip(a, b):
        assert np.array_equal(fa.det_boxes, fb.det_boxes)
        assert len(fa.gt_boxes) == 3 and len(fa.gt_ids) == 3


def test_miss_rate_drops_detections_but_never_ground_truth():
    from DeepDataMiningLearning.ngperception.tracking.datasets import synthetic_sequence
    frames = synthetic_sequence(n_frames=40, n_objects=4, miss_rate=0.5, seed=3)
    assert all(len(f.gt_boxes) == 4 for f in frames)
    total_det = sum(len(f.det_boxes) for f in frames)
    assert 0 < total_det < 40 * 4


# ------------------------------------------------------ MOTSequence end to end
def _write_seq(root, name, gt_rows, det_rows):
    import os
    d = root / name
    (d / "gt").mkdir(parents=True)
    (d / "det").mkdir(parents=True)
    (d / "gt" / "gt.txt").write_text("".join(gt_rows), encoding="utf-8")
    (d / "det" / "det.txt").write_text("".join(det_rows), encoding="utf-8")
    return str(root)


def test_mot_sequence_reads_gt_and_det_in_frame_order(tmp_path):
    from DeepDataMiningLearning.ngperception.tracking.datasets import MOTSequence
    root = _write_seq(
        tmp_path, "SEQ-01",
        ["1,1,10,20,30,40,1,-1,-1,-1\n", "2,1,12,22,30,40,1,-1,-1,-1\n",
         "3,1,14,24,30,40,1,-1,-1,-1\n"],
        ["1,-1,11,21,30,40,0.9,-1,-1,-1\n", "3,-1,15,25,30,40,0.8,-1,-1,-1\n"])
    frames = list(MOTSequence(root, "SEQ-01"))
    assert [f.frame_id for f in frames] == [0, 1, 2]
    assert len(MOTSequence(root, "SEQ-01")) == 3
    assert [len(f.gt_boxes) for f in frames] == [1, 1, 1]
    # frame 1 has ground truth but no detection -- the tracker must still see it
    assert [len(f.det_boxes) for f in frames] == [1, 0, 1]
    assert frames[0].gt_boxes[0].tolist() == [10.0, 20.0, 40.0, 60.0]
    assert frames[0].det_scores[0] == pytest.approx(0.9)


def test_mot_sequence_survives_a_missing_det_file(tmp_path):
    from DeepDataMiningLearning.ngperception.tracking.datasets import MOTSequence
    d = tmp_path / "SEQ-02" / "gt"
    d.mkdir(parents=True)
    (d / "gt.txt").write_text("1,1,0,0,10,10,1,-1,-1,-1\n", encoding="utf-8")
    frames = list(MOTSequence(str(tmp_path), "SEQ-02"))
    assert len(frames) == 1 and len(frames[0].det_boxes) == 0


def test_mot_csv_skips_blank_and_comment_lines(tmp_path):
    from DeepDataMiningLearning.ngperception.tracking.datasets import read_mot_csv
    p = tmp_path / "gt.txt"
    p.write_text("# header\n\n1,1,10,20,30,40,1,-1,-1,-1\n\n", encoding="utf-8")
    assert sorted(read_mot_csv(str(p))) == [0]


def test_sequence_runs_end_to_end_through_tracker_and_evaluator(tmp_path):
    """The path a user actually takes: read a sequence, track it, score it."""
    from DeepDataMiningLearning.ngperception.tracking.datasets import MOTSequence
    gt, det = [], []
    for f in range(1, 9):                                   # 1-indexed on disk
        for oid, x0 in ((1, 10), (2, 200)):
            x = x0 + 6 * (f - 1) * (1 if oid == 1 else -1)
            row = f"{f},{oid},{x},{50 * oid},40,40,1,-1,-1,-1\n"
            gt.append(row)
            det.append(f"{f},-1,{x},{50 * oid},40,40,0.9,-1,-1,-1\n")
    root = _write_seq(tmp_path, "SEQ-03", gt, det)

    tracker = build_tracker("sort", min_hits=1, max_age=2)
    ev = MOTEvaluator()
    ev.new_sequence()
    for fr in MOTSequence(root, "SEQ-03"):
        res = tracker.update(det_from_frame(fr))
        ev.add(fr.gt_ids.tolist(), fr.gt_boxes, res.track_ids.tolist(), res.boxes)
    s = ev.summarize(verbose=False)
    assert s["GT"] == 16 and s["IDSW"] == 0
    assert s["MOTA"] > 0.99 and s["IDF1"] > 0.99


def det_from_frame(frame):
    n = len(frame.det_boxes)
    return Detection(boxes=frame.det_boxes, scores=frame.det_scores,
                     labels=np.zeros(n, np.int64), names=["object"] * n)


# ------------------------------------------------------------ error branches
def test_evaluator_rejects_misaligned_input():
    ev = MOTEvaluator()
    with pytest.raises(ValueError, match="index-aligned"):
        ev.add([1, 2], np.zeros((1, 4), np.float32), [1], np.zeros((1, 4), np.float32))


def test_duplicate_track_ids_in_one_frame_are_rejected():
    bad = TrackResult(track_ids=np.array([5, 5]), boxes=np.zeros((2, 4), np.float32),
                      scores=np.zeros(2, np.float32), labels=np.zeros(2, np.int64),
                      names=["a", "b"])
    with pytest.raises(ValueError, match="duplicate"):
        bad.check_aligned()


def test_base_tracker_is_abstract():
    from DeepDataMiningLearning.ngperception.tracking.trackers.base import BaseTracker
    b = BaseTracker()
    with pytest.raises(NotImplementedError):
        b.update(det([[0, 0, 1, 1]]))
    with pytest.raises(NotImplementedError):
        b.reset()


def test_sort_rejects_an_unknown_variant():
    with pytest.raises(ValueError, match="variant"):
        build_tracker("sort:turbo")


def test_summarize_verbose_prints_without_crashing(capsys):
    ev = MOTEvaluator()
    ev.new_sequence()
    box = np.array([[0, 0, 10, 10]], np.float32)
    ev.add([1], box, [1], box)
    ev.summarize(verbose=True)
    assert "MOTA" in capsys.readouterr().out


def test_a_far_detection_starts_a_new_track_instead_of_hijacking_one():
    """The IoU gate is what stops the assignment matching unrelated boxes.

    Hungarian assignment on its own will happily pair a track with the only
    detection available, however far away it is. Without this case the suite
    passes even when the gate is removed entirely -- a mutation that let any
    detection claim any track went unnoticed until this was added.
    """
    t = build_tracker("sort", min_hits=1, max_age=1)
    first = t.update(det([[0, 0, 40, 40]])).track_ids.tolist()
    assert first == [1]
    # A detection on the far side of the image: IoU with the track is 0.
    res = t.update(det([[900, 900, 940, 940]]))
    ids = res.track_ids.tolist()
    assert 1 not in ids, f"track 1 was hijacked by an unrelated detection: {ids}"
    assert len(ids) == 1 and ids[0] != 1
    assert res.boxes[0][0] > 800, res.boxes[0].tolist()


def test_a_collapsing_box_keeps_a_finite_positive_size():
    """Exercises the area clamp in _KalmanBoxTracker.predict.

    A box shrinking fast gives the filter a negative area velocity; without the
    clamp the predicted area goes through zero and the width/height come back
    as nan (sqrt of a negative), which then poisons every later IoU.
    """
    t = build_tracker("sort", min_hits=1, max_age=40)
    for side in (400, 240, 120, 40, 8, 2):                # collapsing fast
        t.update(det([[100, 100, 100 + side, 100 + side]]))
    last = None
    for _ in range(12):                                   # coast on prediction alone
        t.update(det(np.zeros((0, 4), np.float32)))
        assert t._tracks, "track retired too early for this check"
        last = t._tracks[0].box
        assert np.isfinite(last).all(), f"box went non-finite: {last.tolist()}"
    w, h = last[2] - last[0], last[3] - last[1]
    # With the clamp this stays ~240 px wide; without it the area passes through
    # zero and every corner collapses onto the same point (w = h = 0).
    assert w > 1.0 and h > 1.0, f"box collapsed to a point: w={w} h={h}"
