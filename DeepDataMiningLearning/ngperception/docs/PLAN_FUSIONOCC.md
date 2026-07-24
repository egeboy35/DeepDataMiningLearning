# Plan: FusionOcc-style SOTA fusion occupancy on our BEVFusion

**Goal (user, 2026-07-24):** add a **semantic occupancy head to our BEVFusion** (LiDAR+cam, NDS 0.688,
`bevdet/outputs/mm3dtrain/`) to get a **SOTA-competitive fusion occ** number — answering the earlier
"camera-only 0.163 isn't SOTA-credible" concern. **FusionOcc (MM2024, 56.6 mIoU Occ3D-nuScenes)** is
the reference/recipe.

## FusionOcc setup (done)
- Cloned `Others/FusionOcc` (gitignored); checkpoint (56.6 model, 440 MB) at
  `Others/FusionOcc/ckpts/fusionocc.pth`.
- **Env blocker (same as FlashOcc):** needs torch 1.10.1+cu113, mmcv-full 1.5.3, mmdet 2.25.1 — **can't
  run on H100 (sm_90)**. So we do NOT run FusionOcc natively; we **adapt its recipe onto our modern
  BEVFusion** (mmdet3d 1.4, runs on H100), using its checkpoint/config as reference.

## FusionOcc recipe (from `configs/fusion_occ/fusion_occ.py` + `detectors/fusion_occ.py`)
- **Grid = Occ3D**: `point_cloud_range=[-40,-40,-1,40,40,5.4]`, 18 classes, lidar voxel 0.05 m.
- **Camera branch**: Swin-B → FPN_LSS(256) → **CrossModalLSS** (depth 88 bins, ASPP; depth guided by
  precomputed **image segmentation** `img_seg/` — the "semi-supervised dense depth", 2D-space fusion).
- **LiDAR branch**: `CustomSparseEncoder` (sparse conv, 5→32 ch), **7 adjacent sweeps** aggregated.
- **3D fusion**: concat camera-3D (`numC_Trans`) + LiDAR-3D (32) → **`CustomResNet3D` occ encoder** +
  `LSSFPN3D` neck → occ head (18-cls, CE). Plus an auxiliary `fuse_loss` (weight 0.1).
- Inputs: `img_inputs, points, sparse_depth, segs, voxel_semantics(Occ3D-GT), mask_camera`.

## Adaptation onto our BEVFusion (the build)
Our `bevdet` BEVFusion (mmdet3d 1.4) already produces LiDAR sparse features + a fused BEV. Add:
1. **Occ3D-grid 3D feature**: take the LiDAR voxel features + camera BEV, place on the 200×200×16 @0.4m
   Occ3D grid (BEVFusion's det grid is 180×180 @[-54,54] — build the occ branch on the *occ* grid, as
   FusionOcc does, not the det grid → no resample mismatch).
2. **3D occ encoder + head** (CustomResNet3D-style) → 18-class occ, CE vs Occ3D-GT.
3. (Optional v2) image-seg-guided depth (CrossModalLSS) for the SOTA-grade number.
Train the occ head on our frozen/lightly-finetuned BEVFusion; eval Occ3D mIoU. Target: FusionOcc-grade
(~0.5+ fusion mIoU) — reviewer-competitive.

## Status / next
- [x] FusionOcc cloned, checkpoint downloaded, recipe extracted, env blocker identified.
- [ ] Port/adapt: LiDAR-voxel + camera-BEV → Occ3D-grid 3D occ encoder + head onto our BEVFusion.
- [ ] Train occ head on our BEVFusion (NDS 0.688 detector) → fusion occ mIoU; compare vs FusionOcc 56.6
  and our camera-only LSS backbone benchmark.
- Reference: FusionOcc `mmdet3d/models/{detectors/fusion_occ.py, necks/fusion_view_transformer.py,
  backbones/lidar_encoder.py}` = the modules to port.
