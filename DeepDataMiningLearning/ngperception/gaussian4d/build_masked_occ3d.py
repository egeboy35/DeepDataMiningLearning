"""
gaussian4d.build_masked_occ3d
============================
Decompose the occ->detection transfer benefit: build occ PRETEXT caches from Occ3D-GT keeping only
BACKGROUND (stuff) classes, only FOREGROUND (objects), or ALL. Pretrain an occ student on each ->
det-transfer -> where does the +32% live? (Hypothesis from DynamicOcc negative: dense background,
not sharp foreground.) Uses real Occ3D-GT so it's a clean upper-bound decomposition before any
label-free background work. Output = TeacherTarget labels.npz (semantics + weight=mask_camera).

    python -m ...gaussian4d.build_masked_occ3d --gts <gts> --keep bg  --out-dir <cache_bg>  --n 2044
    python -m ...gaussian4d.build_masked_occ3d --gts <gts> --keep fg  --out-dir <cache_fg>  --n 2044
    python -m ...gaussian4d.build_masked_occ3d --gts <gts> --keep all --out-dir <cache_all> --n 2044
"""
from __future__ import annotations
import argparse, os
import numpy as np
from .teachers.base import TeacherTarget, FREE

FG = set(range(0, 11))          # others(0) + 10 object classes 1..10
BG = set(range(11, 17))         # 11..16 driveable/other_flat/sidewalk/terrain/manmade/vegetation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gts", required=True)
    ap.add_argument("--keep", required=True, choices=["bg", "fg", "all"])
    ap.add_argument("--out-dir", required=True); ap.add_argument("--n", type=int, default=2044)
    ap.add_argument("--labelgen-cache", default=None, help="if set, match its token set (2044)")
    args = ap.parse_args()
    from ..occupancy.datasets import Occ3DNuScenesDataset
    occ = Occ3DNuScenesDataset(args.gts, scenes=None)
    items = occ.items
    if args.labelgen_cache:
        items = [(sc, tok, lp) for sc, tok, lp in items
                 if os.path.isfile(os.path.join(args.labelgen_cache, tok + ".npz"))]
    items = items[: args.n]
    os.makedirs(args.out_dir, exist_ok=True)
    keep = {"bg": BG, "fg": FG, "all": FG | BG}[args.keep]
    print(f"[masked] keep={args.keep} ({len(keep)} classes) on {len(items)} frames -> {args.out_dir}", flush=True)
    done = 0
    for i, (sc, tok, lp) in enumerate(items):
        outp = os.path.join(args.out_dir, tok + ".npz")
        if os.path.isfile(outp):
            continue
        g = np.load(lp)
        sem = g["semantics"].astype(np.uint8).copy()
        mask = ~np.isin(sem, list(keep)) & (sem != FREE)     # occupied voxels of dropped classes
        sem[mask] = FREE                                      # -> free (not supervised as object)
        weight = g["mask_camera"].astype(np.float32)          # supervise camera-visible voxels
        TeacherTarget(sem, weight).save(outp); done += 1
        if done % 300 == 0:
            print(f"  {done}/{len(items)}", flush=True)
    print(f"[masked] done: wrote {done}", flush=True)


if __name__ == "__main__":
    main()
