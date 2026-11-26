"""Crop printed-edition pages (e.g. Horner) to the main text above the first horizontal rule.

The rule separating text from apparatus is a long, thin, continuous dark run. We take the
first row (below the top 8% of the page) whose longest dark run exceeds 15% of the page width.
    python scripts/crop_above_rule.py IN_DIR OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for f in sorted(src.glob("*.jpg")):
    im = Image.open(f).convert("L")
    a = np.asarray(im) < 110
    H, W = a.shape
    cut = None
    for y in range(int(0.08 * H), int(0.95 * H)):
        row = a[y]
        # longest run of dark pixels in this row
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        if len(starts) and (ends - starts).max() > 0.15 * W:
            cut = y
            break
    if cut is None or cut < 0.15 * H:
        print("no rule", f.name)
        continue
    im.crop((0, 0, W, cut - 5)).convert("RGB").save(dst / f.name)
    print(f.name, "cut at", cut, "/", H)
