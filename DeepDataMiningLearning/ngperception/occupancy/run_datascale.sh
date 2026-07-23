#!/usr/bin/env bash
# Data-scaling test: is the occ->detection transfer benefit driven by PRETRAINING DATA AMOUNT?
# Train the Occ3D-GT occ pretext (train_lss, same trainer as lss_occ_full) at 2044 & 8000 frames ->
# det-transfer @2k seed1. Curve: 2044 -> 8000 -> 28k(lss_occ_full=.163). If it climbs with data, the
# label-free nulls (all at 2044 frames) were DATA STARVATION, not a fundamental label-free limit.
set -e
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.bashrc 2>/dev/null || true; conda activate py310 2>/dev/null || true
ROOT=/data/rnd-liu/Datasets/nuScenes
NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
OUT=DeepDataMiningLearning/ngperception/output
CSV=$OUT/label_eff/results_datascale.csv
PY="python -m DeepDataMiningLearning.ngperception"
OCFG="--backbone dinov2_base --decoder-layers 4 --decoder-hidden 96 --refine-iters 1"
DCFG="--backbone dinov2_base --decoder-layers 4 --decoder-hidden 96 --refine-iters 1 --det-head center"
mkdir -p $OUT/label_eff
[ -f $CSV ] || echo "pretrain_frames,budget,seed,mAP,NDS,ped_AP" > $CSV

for N in 2044 8000; do
  ST=$OUT/occgt_${N}
  # 1) occ pretrain on Occ3D-GT with N frames (train_lss reads gts directly)
  [ -f $ST/lss_occ.pth ] || $PY.occupancy.train_lss --nusc $NUSC --gts $GTS \
      --max-samples $N --val-samples 200 --epochs 24 --batch-size 2 --lr 2e-3 $OCFG --out-dir $ST
  # 2) det-transfer @2k seed1
  grep -q "^${N},2000,1," $CSV && continue
  tag=ds_${N}_b2000
  $PY.occupancy.train_det_ablation --nusc $NUSC --gts $GTS --pretrained $ST/lss_occ.pth $DCFG \
      --max-samples 2000 --val-samples 200 --epochs 12 --batch-size 8 --lr 2e-3 --cosine \
      --num-workers 8 --seed 1 --out-dir $OUT/label_eff/$tag
  $PY.occupancy.eval_det_ablation_official --nusc $NUSC --gts $GTS \
      --ckpt $OUT/label_eff/$tag/det_abl.pth --out-dir $OUT/label_eff/${tag}_eval > $OUT/label_eff/${tag}_eval.log 2>&1 || true
  mAP=$(grep -oE "mAP = [0-9.]+" $OUT/label_eff/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+")
  NDS=$(grep -oE "NDS = [0-9.]+" $OUT/label_eff/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+")
  ped=$(grep -E "pedestrian" $OUT/label_eff/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+" | tail -1)
  echo "${N},2000,1,${mAP:-NA},${NDS:-NA},${ped:-NA}" >> $CSV
  echo "[ds] pretrain=$N -> det mAP=${mAP} NDS=${NDS}"
  rm -f $OUT/label_eff/$tag/det_abl.pth
done
echo "[ds] DONE -> $CSV  (compare to lss_occ_full@28k = 0.163)"
