"""
ngperception.occupancy.evaluator
================================

The official **Occ3D-nuScenes** metric: mean IoU over the 17 occupied semantic classes,
evaluated **only on camera-visible voxels** (`mask_camera`). We also report a class-
agnostic **geometric IoU** (occupied-vs-free), which is what a pure depth->voxel baseline
can hope to do well even with no semantics.

Grid: x,y ∈ [-40, 40] m, z ∈ [-1, 5.4] m, voxel 0.4 m → (200, 200, 16). Class 17 = free.

IoU per class c = TP_c / (TP_c + FP_c + FN_c), accumulated over the dataset (not averaged
per-frame), then mIoU = mean over classes 0..16.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np

# 18 Occ3D-nuScenes classes; index 17 is free space (excluded from mIoU).
OCC3D_CLASSES = [
    "others", "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade",
    "vegetation", "free",
]
FREE = 17
NUM_SEMANTIC = 17                      # classes 0..16 count toward mIoU

# Published Occ3D-nuScenes *val* mIoU (camera-only) — the SOTA target our runnable
# depth-lift baseline is measured against. These are cited reference numbers (the learned
# nets need mmdetection3d, not runnable here on torch 2.10), NOT re-evaluated locally.
SOTA_REFERENCE = {
    "MonoScene (CVPR'22)":       6.06,
    "BEVFormer (occ)":           26.88,
    "CTF-Occ (Occ3D, NeurIPS'23)": 28.53,
    "FlashOcc (2023)":           32.0,
    "Dr.Occ (depth-guided)":     43.43,
    "EFFOcc (18.4M params)":     50.46,
}


def print_sota_reference():
    print("  --- published Occ3D-nuScenes val mIoU (camera-only, cited) ---")
    for k, v in SOTA_REFERENCE.items():
        print(f"    {k:32s} {v:.2f}")

# grid geometry (Occ3D-nuScenes defaults)
PC_RANGE = np.array([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4])
VOXEL_SIZE = 0.4
GRID_SIZE = np.array([200, 200, 16])   # (X, Y, Z)


class OccupancyEvaluator:
    """Accumulates a global confusion over camera-visible voxels and reports IoU/mIoU."""

    def __init__(self, use_camera_mask: bool = True):
        self.use_camera_mask = use_camera_mask
        # per-class TP/FP/FN over the 17 semantic classes
        self.tp = np.zeros(NUM_SEMANTIC, np.int64)
        self.fp = np.zeros(NUM_SEMANTIC, np.int64)
        self.fn = np.zeros(NUM_SEMANTIC, np.int64)
        # class-agnostic occupied-vs-free
        self.g_tp = self.g_fp = self.g_fn = 0
        self.n = 0

    def add(self, pred: np.ndarray, gt: np.ndarray, mask_camera: np.ndarray = None):
        """pred, gt: (200,200,16) uint8 semantic grids; mask_camera: bool/uint8 grid."""
        if self.use_camera_mask and mask_camera is not None:
            sel = mask_camera.astype(bool)
        else:
            sel = np.ones_like(gt, bool)
        p, g = pred[sel].astype(np.int64), gt[sel].astype(np.int64)

        for c in range(NUM_SEMANTIC):
            pc, gc = (p == c), (g == c)
            self.tp[c] += int(np.sum(pc & gc))
            self.fp[c] += int(np.sum(pc & ~gc))
            self.fn[c] += int(np.sum(~pc & gc))
        # geometric: occupied = any class != free
        po, go = (p != FREE), (g != FREE)
        self.g_tp += int(np.sum(po & go))
        self.g_fp += int(np.sum(po & ~go))
        self.g_fn += int(np.sum(~po & go))
        self.n += 1

    def summarize(self, verbose: bool = True) -> Dict[str, float]:
        union = self.tp + self.fp + self.fn
        # A class absent from both the ground truth and the prediction has an
        # undefined IoU (0/0). Averaging it in as 0 divides by the full class
        # count instead of the classes actually present, which scales mIoU by
        # n_present / NUM_SEMANTIC -- a perfect prediction on a scene holding 3
        # of the 17 classes would score 0.176. Occ3D-nuScenes averages over the
        # present classes (np.nanmean), so absent classes are excluded here too.
        present = union > 0
        iou = np.where(present, self.tp / np.maximum(union, 1), np.nan)
        miou = float(np.nanmean(iou)) if present.any() else 0.0
        geo = self.g_tp / max(self.g_tp + self.g_fp + self.g_fn, 1)
        out = {"mIoU": miou, "geo_IoU": float(geo), "num_samples": self.n,
               "classes_present": int(present.sum()), "num_classes": int(NUM_SEMANTIC)}
        # per_class keeps a plain float per class; absent classes report nan so a
        # reader can tell "the model missed it" from "it was never there".
        out["per_class"] = {OCC3D_CLASSES[c]: float(iou[c]) for c in range(NUM_SEMANTIC)}
        if verbose:
            print(f"  samples={self.n}  mIoU={miou:.3f}  geometric IoU={geo:.3f}"
                  f"  ({int(present.sum())}/{int(NUM_SEMANTIC)} classes present)")
            # nan compares False against everything, so absent classes have to be
            # dropped before sorting or they land in arbitrary positions.
            scored = [(k, v) for k, v in out["per_class"].items() if not np.isnan(v)]
            top = sorted(scored, key=lambda x: -x[1])[:6]
            print("  best classes: " + "  ".join(f"{k}={v:.2f}" for k, v in top))
        return out


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
#   python -m DeepDataMiningLearning.ngperception.occupancy.evaluator
# Synthetic check: predicting GT scores mIoU=1.0; predicting all-free scores 0.
# ===========================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 18, size=(200, 200, 16)).astype(np.uint8)
    mask = rng.random((200, 200, 16)) < 0.3
    ev = OccupancyEvaluator()
    ev.add(gt.copy(), gt, mask); print("perfect:"); ev.summarize()
    ev2 = OccupancyEvaluator()
    ev2.add(np.full_like(gt, FREE), gt, mask); print("all-free:"); ev2.summarize()

    # A scene need not contain all 17 classes. A perfect prediction must still
    # score 1.0 -- averaging the absent classes in as 0 would report
    # n_present / NUM_SEMANTIC instead.
    sparse = np.zeros((40, 40, 8), np.uint8)
    for c in range(1, 4):
        sparse[(c - 1) * 2:(c - 1) * 2 + 2] = c
    ev3 = OccupancyEvaluator()
    ev3.add(sparse.copy(), sparse)
    print("perfect on a scene with few classes:")
    m = ev3.summarize()
    assert abs(m["mIoU"] - 1.0) < 1e-9, m["mIoU"]
    assert m["classes_present"] < m["num_classes"], m
    print("OK")
