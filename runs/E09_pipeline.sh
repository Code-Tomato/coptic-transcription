#!/bin/bash
# E09: pseudo-label round 3 over pool1 (442 pages) + pool2 (1306 pages) with E08, then train.
set -e
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
for i in 1 2; do  # second pass retries any page that failed
  $PY -I scripts/segment_pages.py --pages data/raw/vatlib/pool2_700 --out data/pool2_lines --model models/segmenter.safetensors > data/pool2_seg$i.log 2>&1
done
$PY -I scripts/pseudo_label.py score --ckpt runs/E08_gold3x_pseudo2a_synth30/best.pt --lm data/lm/cs_order5.pkl --lines data/vatlib_lines --out data/pseudo/round3_pool1_E08.tsv > data/pseudo/round3_pool1.log 2>&1
$PY -I scripts/pseudo_label.py score --ckpt runs/E08_gold3x_pseudo2a_synth30/best.pt --lm data/lm/cs_order5.pkl --lines data/pool2_lines --out data/pseudo/round3_pool2_E08.tsv > data/pseudo/round3_pool2.log 2>&1
$PY -I scripts/select_pseudo.py data/pseudo/round3_pool1_E08.tsv data/pseudo/round3 --min-conf 0.8 > data/pseudo/round3_select.log
cp data/pseudo/round3/train.lst data/pseudo/round3/pool1.lst
$PY -I scripts/select_pseudo.py data/pseudo/round3_pool2_E08.tsv data/pseudo/round3 --min-conf 0.8 >> data/pseudo/round3_select.log
cat data/pseudo/round3/pool1.lst data/pseudo/round3/train.lst > data/pseudo/round3/all_pseudo.lst
for k in $(seq 10); do cat data/scam_a/train.lst; done > data/pseudo/round3/gold10x.lst
cat data/pseudo/round3/gold10x.lst data/pseudo/round3/all_pseudo.lst > data/pseudo/round3/train_plus_gold10x.lst
wc -l data/pseudo/round3/*.lst >> data/pseudo/round3_select.log
$PY -I scripts/train_crnn.py --name E09_gold10x_pseudo3_pool12_synth30 --train data/pseudo/round3/train_plus_gold10x.lst \
  --synth data/synth/v1 --synth-ratio 0.3 --steps 20000 --eval-every 1000 --workers 6 > runs/E09.log 2>&1
