#!/usr/bin/env bash
# Detection arm of the backbone benchmark: det-transfer each occ-pretrained backbone (from
# run_backbone_bench.sh) -> official nuScenes mAP @2k. Combined with occ_results.csv this gives the
# OCC+DET ranking. Also answers: does train_lss's strong occ @2044 (DINOv2-B .288 / -L .316) yield
# good DET (~lss_occ_full .163)? -> if yes, the earlier det gap was the train_student recipe, not data.
# Runs in parallel with the occ sweep (waits for each occ ckpt). VGGT-det needs cache plumbing -> skipped here.
set +e
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.bashrc 2>/dev/null || true; conda activate py310 2>/dev/null || true
ROOT=/data/rnd-liu/Datasets/nuScenes
NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
OUT=DeepDataMiningLearning/ngperception/output/backbone_bench
CSV=$OUT/det_results.csv
PY="python -m DeepDataMiningLearning.ngperception.occupancy"
DCFG="--decoder-layers 4 --decoder-hidden 96 --refine-iters 1 --det-head center"
mkdir -p $OUT
[ -f $CSV ] || echo "backbone,budget,seed,mAP,NDS,ped_AP" > $CSV

for bb in dinov2_base dinov2_large radio siglip2; do        # VGGT-det deferred (needs cache plumbing)
  grep -q "^${bb}," $CSV && { echo "[det] skip $bb"; continue; }
  ST=$OUT/occ_${bb}/lss_occ.pth
  echo "[det] waiting for occ ckpt: $ST"
  until [ -f "$ST" ]; do sleep 60; done
  sleep 30                                           # let the occ trainer fully release the file
  tag=det_${bb}
  $PY.train_det_ablation --nusc $NUSC --gts $GTS --pretrained $ST --backbone $bb $DCFG \
      --max-samples 2000 --val-samples 200 --epochs 12 --batch-size 8 --lr 2e-3 --cosine \
      --num-workers 6 --seed 1 --out-dir $OUT/$tag
  $PY.eval_det_ablation_official --nusc $NUSC --gts $GTS \
      --ckpt $OUT/$tag/det_abl.pth --out-dir $OUT/${tag}_eval > $OUT/${tag}_eval.log 2>&1
  mAP=$(grep -oE "mAP = [0-9.]+" $OUT/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+")
  NDS=$(grep -oE "NDS = [0-9.]+" $OUT/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+")
  ped=$(grep -E "pedestrian" $OUT/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+" | tail -1)
  echo "${bb},2000,1,${mAP:-NA},${NDS:-NA},${ped:-NA}" >> $CSV
  echo "[det] $bb -> mAP=${mAP} NDS=${NDS}"
  rm -f $OUT/$tag/det_abl.pth
done
echo "[det] DONE -> $CSV"
