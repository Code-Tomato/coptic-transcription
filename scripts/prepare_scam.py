"""Cut SCAM page images into line images using the CVAT polygons.

Output layout (Kraken "path" format, also used by our own trainers):
    <out>/<split>/<page>__<NNN>.png
    <out>/<split>/<page>__<NNN>.gt.txt
plus <out>/manifest.tsv with split, image path and normalized text.

The CVAT annotations were drawn on half-resolution images, so polygon
coordinates are rescaled to the size of the downloaded image.
"""
import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from textnorm import normalize  # noqa: E402

SPLIT_NAMES = {"train": "train", "eval": "val", "test": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=float, default=0.12, help="vertical padding as fraction of line height")
    args = ap.parse_args()

    out = Path(args.out)
    rows = []
    root = ET.parse(args.xml).getroot()
    for im in root.iter("image"):
        name = im.get("name")
        split = SPLIT_NAMES[im.get("subset")]
        img_path = Path(args.images) / f"{name}.jpg"
        if not img_path.exists():
            print("missing image", img_path, file=sys.stderr)
            continue
        page = Image.open(img_path).convert("RGB")
        sx = page.width / float(im.get("width"))
        sy = page.height / float(im.get("height"))
        (out / split).mkdir(parents=True, exist_ok=True)
        polys = sorted(im.iter("polygon"), key=lambda p: p.get("label"))
        for p in polys:
            content = [a.text or "" for a in p.iter("attribute") if a.get("name") == "Content"]
            text = normalize(content[0] if content else "")
            if not text:
                continue
            pts = [tuple(map(float, xy.split(","))) for xy in p.get("points").split(";")]
            xs = [x * sx for x, _ in pts]
            ys = [y * sy for _, y in pts]
            h = max(ys) - min(ys)
            x0 = max(0, int(min(xs) - 0.05 * h))
            x1 = min(page.width, int(max(xs) + 0.05 * h))
            y0 = max(0, int(min(ys) - args.pad * h))
            y1 = min(page.height, int(max(ys) + args.pad * h))
            line = page.crop((x0, y0, x1, y1))
            idx = p.get("label").split("_")[-1]
            stem = f"{split}/{name.replace('.', '-')}__{idx}"
            line.save(out / f"{stem}.png")
            (out / f"{stem}.gt.txt").write_text(text, encoding="utf-8")
            rows.append((split, f"{stem}.png", text))

    with open(out / "manifest.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(["split", "image", "text"])
        w.writerows(rows)
    from collections import Counter
    print(Counter(r[0] for r in rows))


if __name__ == "__main__":
    main()
