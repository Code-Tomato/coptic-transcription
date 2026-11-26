"""Segment page images into text lines with Kraken's default BLLA model and crop them.

Crops follow prepare_scam.py: the bounding box of the line polygon, with vertical
padding as a fraction of line height, so pseudo-labelled lines look like SCAM crops.

    python scripts/segment_pages.py --pages data/raw/vatlib/pages --out data/vatlib_lines
Writes <out>/<page>__<NNN>.png and <out>/segmentation.jsonl (one record per page).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def column_order(boxes, page_width):
    """Reading order for multi-column pages: cluster lines into columns by their
    horizontal centre (split at gaps wider than 12% of the page), then read each
    column top to bottom, columns left to right. Returns a list of indices."""
    if not boxes:
        return []
    centers = sorted(range(len(boxes)), key=lambda i: (boxes[i][0] + boxes[i][2]) / 2)
    cols, cur = [], [centers[0]]
    for a, b in zip(centers, centers[1:]):
        ca = (boxes[a][0] + boxes[a][2]) / 2
        cb = (boxes[b][0] + boxes[b][2]) / 2
        if cb - ca > 0.12 * page_width:
            cols.append(cur)
            cur = []
        cur.append(b)
    cols.append(cur)
    cols.sort(key=lambda c: min(boxes[i][0] for i in c))
    return [i for c in cols for i in sorted(c, key=lambda i: boxes[i][1])]


def fast_polygons(baselines, im_feats=None, suppl_obj=None, topline=False, raise_on_error=False, **kw):
    """Replacement for kraken's slow seam-carving polygonizer: a box around the
    baseline whose height comes from the distance to the nearest line above."""
    out = []
    others = [np.asarray(o, dtype=float) for o in (suppl_obj or []) if len(o) >= 2]
    for bl in baselines:
        bl = np.asarray(bl, dtype=float)
        x0, x1 = bl[:, 0].min(), bl[:, 0].max()
        y = float(np.median(bl[:, 1]))
        dists = []
        for o in others:
            if o[:, 0].max() < x0 or o[:, 0].min() > x1:
                continue
            oy = float(np.median(o[:, 1]))
            if abs(oy - y) > 3:
                dists.append(abs(oy - y))
        d = min(dists) if dists else 40.0
        top = [(float(x), float(yy) - 0.80 * d) for x, yy in bl]
        bot = [(float(x), float(yy) + 0.18 * d) for x, yy in bl[::-1]]
        out.append(np.array(top + bot))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", required=True, nargs="+", help="dirs or image files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=float, default=0.12)
    ap.add_argument("--max-side", type=int, default=2000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", help="segmentation model (default: kraken's BLLA)")
    ap.add_argument("--exact-polygons", action="store_true", help="use kraken's slow polygonizer")
    args = ap.parse_args()
    if not args.exact_polygons:
        import kraken.lib.vgsl.spred as spred
        spred.calculate_polygonal_environment = fast_polygons
        args.pad = 0.0

    from kraken.configs import SegmentationInferenceConfig
    from kraken.tasks import SegmentationTaskModel

    model = SegmentationTaskModel.load_model(args.model) if args.model else SegmentationTaskModel.load_model()
    try:
        model = model.to(args.device)
    except Exception:
        pass
    cfg = SegmentationInferenceConfig(accelerator="cpu") if args.device == "cpu" else SegmentationInferenceConfig()
    try:
        cfg = SegmentationInferenceConfig(accelerator="gpu", device=[0]) if args.device.startswith("cuda") else cfg
    except TypeError:
        pass

    files = []
    for p in args.pages:
        p = Path(p)
        files += sorted(p.glob("*.jpg")) if p.is_dir() else [p]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    seg_log = out / "segmentation.jsonl"
    if seg_log.exists():
        done = {json.loads(l)["page"] for l in open(seg_log)}
    with open(seg_log, "a") as log:
        for i, f in enumerate(files):
            if f.stem in done:
                continue
            im = Image.open(f).convert("RGB")
            if max(im.size) > args.max_side:
                im.thumbnail((args.max_side, args.max_side))
            try:
                with torch.inference_mode():
                    seg = model.predict(im, cfg)
            except Exception as e:  # keep going on odd pages
                print("FAIL", f.name, e, flush=True)
                continue
            lines = []
            cand = []
            for ln in seg.lines:
                xs = [x for x, _ in ln.boundary]
                ys = [y for _, y in ln.boundary]
                h = max(ys) - min(ys)
                if h < 8 or (max(xs) - min(xs)) < 2 * h:
                    continue  # drop specks before ordering: they can bridge the column gap
                cand.append((min(xs), min(ys), max(xs), max(ys), ln))
            ordered = [cand[k][4] for k in column_order([c[:4] for c in cand], im.width)]
            for j, ln in enumerate(ordered):
                xs = [x for x, _ in ln.boundary]
                ys = [y for _, y in ln.boundary]
                h = max(ys) - min(ys)
                if h < 8 or (max(xs) - min(xs)) < 2 * h:
                    continue
                x0, x1 = max(0, int(min(xs) - 0.05 * h)), min(im.width, int(max(xs) + 0.05 * h))
                y0, y1 = max(0, int(min(ys) - args.pad * h)), min(im.height, int(max(ys) + args.pad * h))
                name = f"{f.stem.replace('.', '-')}__{j:03d}.png"
                im.crop((x0, y0, x1, y1)).save(out / name)
                lines.append({"file": name, "bbox": [x0, y0, x1, y1], "baseline": ln.baseline})
            log.write(json.dumps({"page": f.stem, "size": im.size, "lines": lines}) + "\n")
            log.flush()
            if i % 20 == 0:
                print(i, f.name, len(lines), flush=True)


if __name__ == "__main__":
    main()
