#!/bin/bash
# Scoring step of cloud_C1.sh, re-run after eval_suite.py crashed (too many open files; fixed).
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
PY="python -I"
$PY scripts/eval_suite.py models/recognizer.pt --out runs/_cloud_check_E12 --workers 48 > runs/_cloud_check_E12.log 2>&1
for r in E13_all_r34 E14_all_r50_h512 E13b_ft_mix E14b_ft_mix; do
  $PY scripts/eval_suite.py runs/$r/best.pt --workers 48 > runs/$r/eval_suite.log 2>&1
done
echo ALLDONE > runs/cloud_C1.done
