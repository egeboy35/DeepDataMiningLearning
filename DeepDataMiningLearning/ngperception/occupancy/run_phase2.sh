#!/usr/bin/env bash
# PHASE 2: light-finetune the winning backbone (DINOv2-large) for the competitive absolute occ+det
# numbers. Unfreeze the FM at a low LR (1e-5), more data (8000), then det-transfer. Compare to the
# frozen-probe (occ .316 / det .114 @2044).
set +e
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.bashrc 2>/dev/null || true; conda activate py310 2>/dev/null || true
ROOT=/data/rnd-liu/Datasets/nuScenes
NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
OUT=DeepDataMiningLearning/ngperception/output
ST=$OUT/ft_dinov2_large
PY="python -m DeepDataMiningLearning.ngperception.occupancy"
CFG="--backbone dinov2_large --decoder-layers 4 --decoder-hidden 96 --refine-iters 1"

# 1) light-finetune occ (unfrozen backbone @ low LR, 8000 frames)
[ -f $ST/lss_occ.pth ] || $PY.train_lss --nusc $NUSC --gts $GTS $CFG \
    --finetune-backbone --backbone-lr 1e-5 --lr 2e-3 --max-samples 8000 --val-samples 300 \
    --epochs 24 --batch-size 2 --amp --cosine --out-dir $ST
miou=$(grep -oE "mIoU[ =:]+[0-9.]+" $ST.log 2>/dev/null | grep -oE "[0-9.]+" | sort -g | tail -1)
echo "[p2] finetuned DINOv2-large occ mIoU=${miou}"

# 2) det-transfer the finetuned backbone @2k (same protocol as the frozen probe)
$PY.train_det_ablation --nusc $NUSC --gts $GTS --pretrained $ST/lss_occ.pth $CFG --det-head center \
    --max-samples 2000 --val-samples 200 --epochs 12 --batch-size 8 --lr 2e-3 --cosine \
    --num-workers 6 --seed 1 --out-dir $OUT/backbone_bench/p2_det
$PY.eval_det_ablation_official --nusc $NUSC --gts $GTS --ckpt $OUT/backbone_bench/p2_det/det_abl.pth \
    --out-dir $OUT/backbone_bench/p2_det_eval > $OUT/backbone_bench/p2_det_eval.log 2>&1
mAP=$(grep -oE "mAP = [0-9.]+" $OUT/backbone_bench/p2_det_eval.log | tail -1 | grep -oE "[0-9.]+")
NDS=$(grep -oE "NDS = [0-9.]+" $OUT/backbone_bench/p2_det_eval.log | tail -1 | grep -oE "[0-9.]+")
echo "[p2] finetuned DINOv2-large -> occ mIoU=${miou} | det mAP=${mAP} NDS=${NDS}" | tee -a $OUT/backbone_bench/phase2.txt
