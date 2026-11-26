"""End-to-end page evaluation: automatic line segmentation + recognition vs gold page text.

For each SCAM-A page in the chosen split, segment the page (segment_pages.py
crops), recognise all lines, join them in reading order and compare with the
gold transcription (all gold lines joined in label order). Line breaks are
ignored: page CER is the edit distance between the two letter streams.

    python scripts/eval_pages.py runs/E02/best.pt --split val [--lm data/lm/cs_order5.pkl]
"""
import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from metrics import edit_distance, graphemes  # noqa: E402
from recognize import Recognizer  # noqa: E402
from textnorm import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPLIT = {"val": "eval", "test": "test", "train": "train"}


def gold_pages(xml, split):
    root = ET.parse(xml).getroot()
    pages = {}
    for im in root.iter("image"):
        if im.get("subset") != SPLIT[split]:
            continue
        lines = []
        for p in sorted(im.iter("polygon"), key=lambda p: p.get("label")):
            t = [a.text or "" for a in p.iter("attribute") if a.get("name") == "Content"]
            t = normalize(t[0] if t else "")
            if t:
                lines.append(t)
        pages[im.get("name")] = lines
    return pages


def letters(s):
    return graphemes("".join(normalize(s).split()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--lm")
    ap.add_argument("--seg", help="segmentation model; default kraken BLLA")
    ap.add_argument("--work", default=str(ROOT / "data/page_eval"))
    args = ap.parse_args()
    pages = gold_pages(ROOT / "data/raw/scam/annotations/page_annotations_RO.xml", args.split)
    work = Path(args.work) / ((Path(args.seg).parent.name + "_") if args.seg else "blla_") / args.split
    pdir = work / "pages"
    pdir.mkdir(parents=True, exist_ok=True)
    for name in pages:
        dst = pdir / f"{name}.jpg"
        if not dst.exists():
            dst.symlink_to((ROOT / "data/raw/scam/images" / f"{name}.jpg").resolve())
    seg = work / "lines" / "segmentation.jsonl"
    if not seg.exists() or len(open(seg).readlines()) < len(pages):
        subprocess.run([str(ROOT / ".venv/bin/python"), "-I", str(ROOT / "scripts/segment_pages.py"),
                        "--pages", str(pdir), "--out", str(work / "lines")] + (["--model", args.seg] if args.seg else []),
                       check=True)
    rec = Recognizer(args.ckpt, lm_path=args.lm)
    tot_err = tot_len = 0
    per_page = {}
    for line in open(seg):
        r = json.loads(line)
        ims = [Image.open(work / "lines" / l["file"]) for l in r["lines"]]
        preds = rec(ims) if ims else []
        hyp, ref = letters("".join(preds)), letters("".join(pages[r["page"]]))
        e = edit_distance(hyp, ref)
        per_page[r["page"]] = {"cer": round(100 * e / len(ref), 2), "lines_found": len(ims),
                               "lines_gold": len(pages[r["page"]])}
        tot_err += e
        tot_len += len(ref)
        (work / f"{r['page']}.pred.txt").write_text("\n".join(preds) + "\n", encoding="utf-8")
    res = {"page_cer": round(100 * tot_err / tot_len, 2), "pages": per_page}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
