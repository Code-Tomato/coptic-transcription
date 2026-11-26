#!/bin/bash
# Cloud round C1 (Vast.ai RTX 3090). Same data for both architectures, all at once from scratch:
#   E13 = current architecture (ResNet34 to layer3 + 2xBiLSTM-256)
#   E14 = bigger (ResNet50 to layer3 + 2xBiLSTM-512)
# data/mix/E13.lst = gold x10 + all 22,384 round-3 pseudo-labels + 526 aligned printed lines x8; synthetic 30%.
# Then each gets the E12 recipe's short mixed fine-tune (E12.lst, lr 2e-4, 4k steps) -> *_ft,
# and every model is scored with eval_suite.py (val, unseen scribes, print).
# The shipped E12 is re-scored first as a check that the cloud box reproduces the local numbers.
set -e
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="python -I"
COMMON="--train data/mix/E13.lst --synth data/synth/v1 --synth-ratio 0.3 --steps 20000 --eval-every 1000 --workers 14"
$PY scripts/train_crnn.py --name E13_all_r34 $COMMON > runs/E13.log 2>&1 &
$PY scripts/train_crnn.py --name E14_all_r50_h512 --backbone resnet50 --hidden 512 $COMMON > runs/E14.log 2>&1 &
sleep 120; $PY scripts/eval_suite.py models/recognizer.pt --out runs/_cloud_check_E12 --workers 32 > runs/_cloud_check_E12.log 2>&1 || true
wait
FT="--train data/mix/E12.lst --synth data/synth/v1 --synth-ratio 0.2 --steps 4000 --lr 2e-4 --warmup 200 --eval-every 500 --workers 14"
$PY scripts/train_crnn.py --name E13b_ft_mix --init runs/E13_all_r34/best.pt $FT > runs/E13b.log 2>&1 &
$PY scripts/train_crnn.py --name E14b_ft_mix --init runs/E14_all_r50_h512/best.pt --backbone resnet50 --hidden 512 $FT > runs/E14b.log 2>&1 &
wait
for r in E13_all_r34 E14_all_r50_h512 E13b_ft_mix E14b_ft_mix; do
  $PY scripts/eval_suite.py runs/$r/best.pt --workers 32 > runs/$r/eval_suite.log 2>&1
done
echo ALLDONE > runs/cloud_C1.done
