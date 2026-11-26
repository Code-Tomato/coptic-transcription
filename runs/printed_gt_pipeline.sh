#!/bin/bash
# Alignment-based ground truth for printed Coptic from Horner vols 3 (John) and 4 (Epistles).
# Never uses vol 2 (Luke), which is the printed benchmark.
set -e
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
PY=.venv/bin/python
until grep -q DONE data/raw/horner/dl34.log; do sleep 30; done
for v in 03 04; do
  $PY -I scripts/crop_above_rule.py data/raw/horner/v$v data/raw/horner/v${v}_main > data/raw/horner/crop_v$v.log
  nice -n 5 $PY -I scripts/segment_pages.py --device cpu --pages data/raw/horner/v${v}_main --out data/horner_v${v}_lines \
      --model models/segmenter.safetensors > data/horner_v${v}_seg.log 2>&1
  CUDA_VISIBLE_DEVICES= $PY -I scripts/predict_pages.py --lines data/horner_v${v}_lines \
      --ckpt runs/E08_gold3x_pseudo2a_synth30/best.pt --lm data/lm/cs_order5.pkl --out data/printed_gt/preds_v$v
  $PY -I scripts/bible_ref.py locate --k 6 --min-hit 0.08 --preds data/printed_gt/preds_v$v --out data/printed_gt/refs_v$v.json > data/printed_gt/locate_v$v.log
  $PY -I scripts/align_printed_gt.py --lines data/horner_v${v}_lines --preds data/printed_gt/preds_v$v \
      --refs data/printed_gt/refs_v$v.json --out data/printed_gt/v$v
done
cat data/printed_gt/v03/train.lst data/printed_gt/v04/train.lst > data/printed_gt/train.lst
wc -l data/printed_gt/train.lst
