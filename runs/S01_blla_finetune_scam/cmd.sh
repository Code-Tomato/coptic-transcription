#!/bin/bash
# S01: fine-tune Kraken's default BLLA baseline segmenter on SCAM-A train pages (16), validate on val (4).
cd "$(dirname "$0")"
../../.venv/bin/ketos -d cuda:0 --precision 16-mixed --workers 4 -s 42 segtrain -f page \
  -i ../../.venv/lib/python3.12/site-packages/kraken/blla.mlmodel --resize new \
  -t ../../data/scam_a_page/train.lst -e ../../data/scam_a_page/val.lst \
  -N 60 -q early --lag 15 --min-epochs 20 \
  -o checkpoints --log-dir logs
