"""Convert SCAM CVAT polygons to PAGE XML (with derived baselines) for `ketos segtrain`.

Baseline: at 10 x-positions we intersect the polygon with a vertical line, take
top/bottom, put the baseline at 78% of the height (Coptic majuscule has few
descenders) and fit a straight line through those points.

    python scripts/scam_to_pagexml.py --xml ... --images ... --out data/scam_a_page
writes <out>/<split>/<page>.xml (+ symlinked image) and <out>/<split>.lst
"""
import argparse
from pathlib import Path
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from shapely.geometry import LineString, Polygon

SPLIT = {"train": "train", "eval": "val", "test": "test"}


def baseline_of(poly: Polygon):
    x0, y0, x1, y1 = poly.bounds
    xs, ys = [], []
    for x in np.linspace(x0 + 0.03 * (x1 - x0), x1 - 0.03 * (x1 - x0), 10):
        inter = poly.intersection(LineString([(x, y0 - 1), (x, y1 + 1)]))
        if inter.is_empty:
            continue
        _, iy0, _, iy1 = inter.bounds
        xs.append(x)
        ys.append(iy0 + 0.78 * (iy1 - iy0))
    if len(xs) < 2:
        return [(x0, y0 + 0.78 * (y1 - y0)), (x1, y0 + 0.78 * (y1 - y0))]
    k, b = np.polyfit(xs, ys, 1)
    return [(x0, k * x0 + b), (x1, k * x1 + b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    lists = {}
    for im in ET.parse(args.xml).getroot().iter("image"):
        name, split = im.get("name"), SPLIT[im.get("subset")]
        src = (Path(args.images) / f"{name}.jpg").resolve()
        W, H = Image.open(src).size
        sx, sy = W / float(im.get("width")), H / float(im.get("height"))
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        img_link = d / f"{name}.jpg"
        if not img_link.exists():
            img_link.symlink_to(src)
        lines = []
        for i, p in enumerate(sorted(im.iter("polygon"), key=lambda p: p.get("label"))):
            pts = [(float(a) * sx, float(b) * sy) for a, b in (xy.split(",") for xy in p.get("points").split(";"))]
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            bl = baseline_of(poly)
            text = [a.text or "" for a in p.iter("attribute") if a.get("name") == "Content"]
            coords = " ".join(f"{int(x)},{int(y)}" for x, y in poly.exterior.coords[:-1])
            blp = " ".join(f"{int(x)},{int(y)}" for x, y in bl)
            lines.append(f'''      <TextLine id="l{i:03d}" custom="structure {{type:default;}}">
        <Coords points="{coords}"/>
        <Baseline points="{blp}"/>
        <TextEquiv><Unicode>{escape(text[0] if text else "")}</Unicode></TextEquiv>
      </TextLine>''')
        xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Metadata><Creator>scam_to_pagexml</Creator><Created>2026-11-21T00:00:00</Created><LastChange>2026-11-21T00:00:00</LastChange></Metadata>
  <Page imageFilename="{name}.jpg" imageWidth="{W}" imageHeight="{H}">
    <TextRegion id="r0" custom="structure {{type:text;}}">
      <Coords points="0,0 {W - 1},0 {W - 1},{H - 1} 0,{H - 1}"/>
{chr(10).join(lines)}
    </TextRegion>
  </Page>
</PcGts>
'''
        (d / f"{name}.xml").write_text(xml, encoding="utf-8")
        lists.setdefault(split, []).append(str((d / f"{name}.xml").resolve()))
    for s, l in lists.items():
        (out / f"{s}.lst").write_text("\n".join(l) + "\n")
        print(s, len(l))


if __name__ == "__main__":
    main()
