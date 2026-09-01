"""What `_match` counts, and what the IoU matrix it built says it should.

The module docstring promises "greedily one-to-one match". These pin that
down: no ground-truth lane is used twice, no prediction is used twice, and a
prediction is not discarded while a ground-truth lane it clears the threshold
against is still free.

Pure numpy, no dataset, no model, no GPU.

    pytest DeepDataMiningLearning/ngperception/lane/tests
"""
import numpy as np
import pytest

from DeepDataMiningLearning.ngperception.lane.culane_metric import (
    CULaneF1, _match, rasterize,
)

H, W = 320, 800                  # CULaneF1's defaults
WIDTH = max(4, round(W / 55))    # its own width rule: 15 px
T = 0.5


def lane(x_top, x_bottom=None, n=40):
    """A vertical-ish lane as the decoder emits it, top to bottom."""
    ys = np.linspace(0, H - 1, n)
    xs = np.linspace(x_top, x_top if x_bottom is None else x_bottom, n)
    return np.stack([xs, ys], axis=1)


def iou_matrix(preds, gts):
    pm = [rasterize(p, H, W, WIDTH) for p in preds]
    gm = [rasterize(g, H, W, WIDTH) for g in gts]
    m = np.zeros((len(pm), len(gm)))
    for i, a in enumerate(pm):
        for j, b in enumerate(gm):
            inter = np.logical_and(a, b).sum()
            union = a.sum() + b.sum() - inter
            m[i, j] = inter / union if union else 0.0
    return m


def best_possible_tp(m, thresh=T):
    """Upper bound on any one-to-one matching, by exhaustive search."""
    from itertools import permutations
    n_p, n_g = m.shape
    best = 0
    if n_p <= n_g:
        for perm in permutations(range(n_g), n_p):
            best = max(best, sum(1 for i, j in enumerate(perm) if m[i, j] >= thresh))
    else:
        for perm in permutations(range(n_p), n_g):
            best = max(best, sum(1 for j, i in enumerate(perm) if m[i, j] >= thresh))
    return best


# ------------------------------------------------------------ the defect
def test_a_prediction_is_not_dropped_while_a_lane_it_matches_is_free():
    """Found by sweeping lane spacing against prediction offset."""
    gts = [lane(300), lane(306)]
    preds = [lane(300), lane(301)]
    m = iou_matrix(preds, gts)

    # the premise: p1's best is the same lane as p0's, and it also clears the
    # threshold against the other lane
    assert m[0].argmax() == 0 and m[0, 0] >= T
    assert m[1].argmax() == 0 and m[1, 1] >= T

    tp, fp, fn = _match(preds, gts, H, W, WIDTH, T)
    assert (tp, fp, fn) == (2, 0, 0)


def test_the_sweep_that_found_it_reports_no_under_counting():
    lost = 0
    for gap in range(4, 18):
        for off in np.arange(0.0, gap, 0.5):
            gts = [lane(300), lane(300 + gap)]
            preds = [lane(300), lane(300 + off)]
            m = iou_matrix(preds, gts)
            tp, _, _ = _match(preds, gts, H, W, WIDTH, T)
            if tp < best_possible_tp(m):
                lost += 1
    assert lost == 0


# ------------------------------------------------ one-to-one is preserved
def test_two_predictions_cannot_claim_one_lane():
    gts = [lane(300)]
    preds = [lane(300), lane(301)]
    tp, fp, fn = _match(preds, gts, H, W, WIDTH, T)
    assert (tp, fp, fn) == (1, 1, 0)


def test_one_prediction_cannot_claim_two_lanes():
    gts = [lane(300), lane(301)]
    preds = [lane(300)]
    tp, fp, fn = _match(preds, gts, H, W, WIDTH, T)
    assert (tp, fp, fn) == (1, 0, 1)


def test_counts_always_add_up():
    rng = np.random.default_rng(3)
    for _ in range(30):
        base = np.sort(rng.uniform(120, 680, size=4))
        gts = [lane(x, x - 90) for x in base]
        preds = [lane(x + rng.normal(0, 8), x - 90 + rng.normal(0, 8)) for x in base]
        tp, fp, fn = _match(preds, gts, H, W, WIDTH, T)
        assert tp + fp == len(preds)
        assert tp + fn == len(gts)
        assert tp <= min(len(preds), len(gts))


# ------------------------------------------------------ threshold respected
def test_a_pair_below_the_threshold_is_never_matched():
    gts = [lane(300)]
    preds = [lane(360)]                      # far enough apart that IoU is 0
    m = iou_matrix(preds, gts)
    assert m[0, 0] < T
    assert _match(preds, gts, H, W, WIDTH, T) == (0, 1, 1)


def test_a_higher_threshold_matches_strictly_fewer():
    gts = [lane(300), lane(306)]
    preds = [lane(300), lane(301)]
    loose = _match(preds, gts, H, W, WIDTH, 0.5)[0]
    tight = _match(preds, gts, H, W, WIDTH, 0.95)[0]
    assert tight <= loose


# ------------------------------------------------------------ empty sides
def test_no_ground_truth_makes_every_prediction_a_false_positive():
    assert _match([lane(300)], [], H, W, WIDTH, T) == (0, 1, 0)


def test_no_predictions_makes_every_lane_a_false_negative():
    assert _match([], [lane(300)], H, W, WIDTH, T) == (0, 0, 1)


def test_both_empty_is_all_zeros():
    assert _match([], [], H, W, WIDTH, T) == (0, 0, 0)


# ------------------------------------------------------- the public class
def test_perfect_predictions_score_one():
    gts = [lane(200, 150), lane(400, 350), lane(600, 550)]
    metric = CULaneF1(img_h=H, img_w=W, iou_thresh=T)
    metric.update([gts], [gts])
    r = metric.compute()
    assert r["tp"] == 3 and r["fp"] == 0 and r["fn"] == 0
    assert r["f1"] == pytest.approx(1.0, abs=1e-6)


def _reference_greedy(m, thresh=T):
    """An independent greedy one-to-one matcher, written from the docstring."""
    pairs = sorted(((m[i, j], i, j) for i in range(m.shape[0])
                    for j in range(m.shape[1]) if m[i, j] >= thresh), reverse=True)
    tp, used_p, used_g = 0, set(), set()
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        tp += 1
        used_p.add(i)
        used_g.add(j)
    return tp


def _argmax_only(m, thresh=T):
    """What the module did before: each prediction gets only its argmax GT."""
    tp, used_g = 0, set()
    for i in np.argsort(-m.max(axis=1)):
        j = int(m[i].argmax())
        if j not in used_g and m[i, j] >= thresh:
            tp += 1
            used_g.add(j)
    return tp


def test_the_class_agrees_with_an_independent_greedy_matcher():
    """Greedy is what the docstring specifies. It is not always the maximum:
    greedy can commit a high pair that blocks two lower ones. The claim here is
    only that the module implements greedy, and that greedy never scores below
    the argmax-only rule it replaces."""
    rng = np.random.default_rng(0)
    metric = CULaneF1(img_h=H, img_w=W, iou_thresh=T)
    greedy_total = optimal_total = argmax_total = 0
    for _ in range(40):
        base = np.sort(rng.uniform(120, 680, size=4))
        gts = [lane(x, x - 90) for x in base]
        preds = []
        for k, x in enumerate(base):
            drift = rng.normal(0, 4)
            if rng.random() < 0.25 and k + 1 < len(base):
                drift += (base[k + 1] - x) * 0.35
            preds.append(lane(x + drift, x - 90 + drift))
        metric.update([preds], [gts])
        m = iou_matrix(preds, gts)
        greedy_total += _reference_greedy(m)
        optimal_total += best_possible_tp(m)
        argmax_total += _argmax_only(m)

    got = metric.compute()["tp"]
    assert got == greedy_total, "the module is not doing greedy one-to-one"
    assert argmax_total <= got <= optimal_total
    assert got > argmax_total, "this sample should show the improvement"
