"""Recognise all lines of already-segmented pages and write <out>/<page>.pred.txt.

    python scripts/predict_pages.py --lines data/heldout_lines --ckpt models/recognizer.pt [--lm ...] --out DIR
Segmentation (segmentation.jsonl from segment_pages.py) is fixed, so different
recognisers are compared on identical line crops.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from recognize import Recognizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lines", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--lm")
ap.add_argument("--out", required=True)
ap.add_argument("--alpha", type=float)
ap.add_argument("--beta", type=float)
a = ap.parse_args()
rec = Recognizer(a.ckpt, lm_path=a.lm, alpha=a.alpha, beta=a.beta)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
pages = [json.loads(l) for l in open(Path(a.lines) / "segmentation.jsonl")]
for r in pages:
    ims = [Image.open(Path(a.lines) / l["file"]) for l in r["lines"]]
    texts = rec(ims) if ims else []
    (out / f"{r['page']}.pred.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
print(len(pages), "pages ->", out)
