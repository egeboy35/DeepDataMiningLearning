#!/usr/bin/env bash
# Chained work after Phase-2 + data-scale: (Dir 2) LiDAR+cam FUSION column, (Dir 4) occ->PLANNING head.
# Extends the backbone benchmark to the full occ + detection + planning stack, single-GPU finetune-budget.
set +e
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.bashrc 2>/dev/null || true; conda activate py310 2>/dev/null || true
ROOT=/data/rnd-liu/Datasets/nuScenes; NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
OUT=DeepDataMiningLearning/ngperception/output; BB=$OUT/backbone_bench
PY="python -m DeepDataMiningLearning.ngperception.occupancy"

# 0) wait for the current jobs to release the GPU
echo "[chain] waiting for Phase-2 + data-scale to finish..."
while pgrep -f "run_phase2|run_datascale_L" >/dev/null; do sleep 120; done
echo "[chain] GPU free -> starting chained work"

# ===== Direction 2: LiDAR+cam FUSION column (DINOv2-large) =====
FCFG="--backbone dinov2_large --decoder-layers 4 --decoder-hidden 96 --refine-iters 1"
FST=$OUT/occ_dinov2L_fusion
[ -f $FST/lss_occ.pth ] || $PY.train_lss --nusc $NUSC --gts $GTS $FCFG --lidar-fusion \
    --max-samples 2044 --val-samples 300 --epochs 24 --batch-size 2 --amp --cosine --out-dir $FST
fmiou=$(grep -oE "mIoU[ =:]+[0-9.]+" $OUT/../occ_dinov2L_fusion.log 2>/dev/null | grep -oE "[0-9.]+" | sort -g | tail -1)
echo "fusion_dinov2_large,${fmiou:-NA},lidar+cam" >> $BB/occ_results.csv
$PY.train_det_ablation --nusc $NUSC --gts $GTS --pretrained $FST/lss_occ.pth $FCFG --lidar-fusion \
    --det-head center --max-samples 2000 --val-samples 200 --epochs 12 --batch-size 8 --lr 2e-3 --cosine \
    --num-workers 6 --seed 1 --out-dir $BB/fusion_det
$PY.eval_det_ablation_official --nusc $NUSC --gts $GTS --ckpt $BB/fusion_det/det_abl.pth \
    --out-dir $BB/fusion_det_eval > $BB/fusion_det_eval.log 2>&1
fmap=$(grep -oE "mAP = [0-9.]+" $BB/fusion_det_eval.log | tail -1 | grep -oE "[0-9.]+")
echo "fusion_dinov2_large,2000,1,${fmap:-NA},lidar+cam" >> $BB/det_results.csv
echo "[chain] FUSION column: occ mIoU=${fmiou} det mAP=${fmap}"

# ===== Direction 4: occ -> PLANNING head (winner DINOv2-large + DINOv2-base reference) =====
[ -f $OUT/ft_dinov2_large/lss_occ.pth ] && PLARGE=$OUT/ft_dinov2_large/lss_occ.pth || PLARGE=$BB/occ_dinov2_large/lss_occ.pth
for entry in "dinov2_large:$PLARGE" "dinov2_base:$BB/occ_dinov2_base/lss_occ.pth"; do
  bb="${entry%%:*}"; ck="${entry##*:}"
  [ -f "$ck" ] || { echo "[chain] plan skip $bb (no ckpt)"; continue; }
  $PY.train_planning --nusc $NUSC --gts $GTS --pretrained $ck --backbone $bb \
      --decoder-layers 4 --decoder-hidden 96 --refine-iters 1 \
      --max-samples 4000 --val-samples 400 --epochs 12 --batch-size 4 --num-workers 6 \
      --out-dir $BB/plan_${bb} > $BB/plan_${bb}.log 2>&1
  res=$(grep -E "\[plan\] epoch" $BB/plan_${bb}.log | tail -1)
  echo "${bb} PLANNING -> ${res}" >> $BB/planning.txt
  echo "[chain] planning $bb -> ${res}"
done
echo "[chain] DONE. occ+det+planning benchmark complete."
