#!/bin/bash
# Cloud round C3. The hardest val lines have ink bleeding through from the other side of the leaf (also common on
# unseen Borg. copt. 109 pages). E17 = E14 recipe + simulated show-through on 30% of real lines (--showthrough 0.3),
# with two seeds, compared against E14/E15 (same recipe without it, seeds 0 and 1).
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="python -I"
COMMON="--train data/mix/E13.lst --synth data/synth/v1 --synth-ratio 0.3 --steps 20000 --eval-every 1000 --workers 14 --backbone resnet50 --hidden 512 --showthrough 0.3"
$PY scripts/train_crnn.py --name E17_E14_showthrough --seed 0 $COMMON > runs/E17.log 2>&1 &
$PY scripts/train_crnn.py --name E17s1_E14_showthrough_seed1 --seed 1 $COMMON > runs/E17s1.log 2>&1 &
wait
for r in E17_E14_showthrough E17s1_E14_showthrough_seed1; do
  $PY scripts/eval_suite.py runs/$r/best.pt --workers 48 > runs/$r/eval_suite.log 2>&1
done
echo ALLDONE > runs/cloud_C3.done
