"""Score any Recognizer (single model or comma-separated ensemble) on a line list.

    python scripts/eval_lines.py "runs/A/best.pt,runs/B/best.pt" data/scam_a/val.lst [--lm models/coptic_char5.pkl]
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from metrics import score  # noqa: E402
from recognize import Recognizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("lst")
ap.add_argument("--lm")
a = ap.parse_args()
paths = [l.strip() for l in open(a.lst) if l.strip()]
rec = Recognizer(a.ckpt, lm_path=a.lm)
preds = rec([Image.open(p) for p in paths])
gts = [Path(p[: p.rfind(".")] + ".gt.txt").read_text(encoding="utf-8") for p in paths]
print(json.dumps({k: round(v, 2) for k, v in score(zip(preds, gts)).items()}))
