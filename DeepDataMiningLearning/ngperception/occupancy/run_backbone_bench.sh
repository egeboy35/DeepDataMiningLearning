#!/usr/bin/env bash
# Backbone benchmark (PHASE 1, frozen-probe): which pretrained backbone gives the best OCCUPANCY on
# nuScenes under a fixed finetune budget? DINOv2 & VGGT are frozen FMs (backbone requires_grad=False /
# cached); ResNet18 trainable baseline. Same LSS lift/head/data (2044 frames = VGGT cache size), so the
# only variable is the backbone. Occ mIoU from train_lss's val eval. (Det arm added once VGGT caches are
# plumbed into the det trainer; DINOv2/ResNet det come from the label-efficiency grids.)
set -e
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.bashrc 2>/dev/null || true; conda activate py310 2>/dev/null || true
ROOT=/data/rnd-liu/Datasets/nuScenes
NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
VF=$ROOT/vggt_feat_cache; VD=$ROOT/vggt_depth_cache
OUT=DeepDataMiningLearning/ngperception/output/backbone_bench
CSV=$OUT/occ_results.csv
PY="python -m DeepDataMiningLearning.ngperception.occupancy.train_lss"
mkdir -p $OUT
[ -f $CSV ] || echo "backbone,occ_mIoU,geo_IoU" > $CSV

for bb in dinov2_base dinov2_large radio vggt; do   # DINOv2-B/L, RADIO (agglomerative FM), VGGT
  grep -q "^${bb}," $CSV && { echo "[bb] skip $bb"; continue; }
  extra=""; [ "$bb" = vggt ] && extra="--vggt-feat-cache $VF --vggt-depth-cache $VD"
  LG=$OUT/occ_${bb}.log
  echo "===== [bb] occ train: $bb ====="
  $PY --nusc $NUSC --gts $GTS --max-samples 2044 --val-samples 300 --epochs 24 --batch-size 2 \
      --lr 2e-3 --backbone $bb --decoder-layers 4 --decoder-hidden 96 --refine-iters 1 \
      $extra --out-dir $OUT/occ_${bb} > $LG 2>&1 || true
  miou=$(grep -oE "mIoU[ =:]+[0-9.]+" $LG | grep -oE "[0-9.]+" | sort -g | tail -1)
  geo=$(grep -oE "geo[_-]?IoU[ =:]+[0-9.]+" $LG | grep -oE "[0-9.]+" | sort -g | tail -1)
  echo "${bb},${miou:-NA},${geo:-NA}" >> $CSV
  echo "[bb] $bb -> occ mIoU=${miou}"
done
echo "[bb] OCC DONE -> $CSV  (ref: lss_occ_full DINOv2 occ mIoU 0.302 on full data)"
