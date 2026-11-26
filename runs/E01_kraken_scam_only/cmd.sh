#!/bin/bash
# E01: Kraken default VGSL (CNN+3xBiLSTM+CTC), SCAM-A train only, augmentation on.
cd "$(dirname "$0")"
../../.venv/bin/ketos -d cuda:0 --workers 6 -s 42 train -f path \
  -t ../../data/scam_a/train.lst -e ../../data/scam_a/val.lst \
  -u NFC -n --augment -B 16 -r 0.001 --lag 30 --min-epochs 50 -N 400 \
  --schedule reduceonplateau --sched-patience 10 \
  -o checkpoints --log-dir logs
