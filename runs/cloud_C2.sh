#!/bin/bash
# Cloud round C2. E14 (ResNet50 + BiLSTM-512, all data at once) was the best all-rounder in C1. Three questions:
#   E15  = E14 with another seed: how much of E14's gain over E13 is noise?
#   E16  = ResNet101 to layer3 + BiLSTM-512, same data: does size keep paying?
#   E14c = E14 fine-tuned on gold only (E10a recipe: lr 1e-4, 2k steps): the in-domain specialist
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="python -I"
COMMON="--train data/mix/E13.lst --synth data/synth/v1 --synth-ratio 0.3 --steps 20000 --eval-every 1000 --workers 10 --hidden 512"
$PY scripts/train_crnn.py --name E15_E14_seed1 --backbone resnet50 --seed 1 $COMMON > runs/E15.log 2>&1 &
$PY scripts/train_crnn.py --name E16_all_r101_h512 --backbone resnet101 $COMMON > runs/E16.log 2>&1 &
$PY scripts/train_crnn.py --name E14c_ft_gold --init runs/E14_all_r50_h512/best.pt --backbone resnet50 --hidden 512 \
  --lr 1e-4 --steps 2000 --warmup 100 --eval-every 250 --workers 10 > runs/E14c.log 2>&1 &
wait
for r in E15_E14_seed1 E16_all_r101_h512 E14c_ft_gold; do
  $PY scripts/eval_suite.py runs/$r/best.pt --workers 48 > runs/$r/eval_suite.log 2>&1
done
echo ALLDONE > runs/cloud_C2.done
