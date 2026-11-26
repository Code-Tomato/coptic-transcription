"""Pseudo-label unlabeled line crops and score their reliability.

    python scripts/pseudo_label.py score  --ckpt models/recognizer.pt --lm models/coptic_char5.pkl \
        --lines data/vatlib_lines --out data/pseudo/round1.tsv
    python scripts/pseudo_label.py calibrate --ckpt ... --lm ... --lst data/scam_a/val.lst

For every line we record:
  text        LM-decoded transcription (what we'd train on)
  greedy      greedy CTC decoding
  min_conf    lowest peak probability over the emitted characters (greedy path)
  mean_conf   mean max-probability over all frames
  agree       greedy == LM decoding
`calibrate` runs the same scoring on gold val lines and prints the CER of the
lines kept at different min_conf thresholds, so the threshold is chosen on data
with known answers.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from metrics import score  # noqa: E402
from recognize import Recognizer  # noqa: E402


def line_stats(prob, codec):
    ids = prob.argmax(0)
    peaks = prob.max(0)
    chars, confs, prev = [], [], 0
    for t, i in enumerate(ids):
        if i != 0 and i != prev:
            chars.append(codec.chars[i - 1])
            confs.append(peaks[t])
        elif i != 0 and i == prev and confs:
            confs[-1] = max(confs[-1], peaks[t])
        prev = i
    return "".join(chars), (min(confs) if confs else 0.0), float(peaks.mean())


_REC = None


def _decode(args):
    p, prob = args
    greedy, mn, mean = line_stats(prob, _REC.codec)
    text = _REC.decode(prob)
    return {"file": str(p), "text": text, "greedy": greedy, "min_conf": round(float(mn), 4),
            "mean_conf": round(mean, 4), "agree": int(text == greedy), "len": len(text)}


def score_images(rec, paths, batch=64, workers=10):
    """GPU forward pass in the main process, LM beam search in a process pool."""
    global _REC
    _REC = rec
    import multiprocessing as mp
    rows = []
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for i in range(0, len(paths), 1024):
            chunk = paths[i:i + 1024]
            probs = []
            for j in range(0, len(chunk), batch):
                probs += rec.probs([Image.open(p) for p in chunk[j:j + batch]])
            rows += pool.map(_decode, list(zip(chunk, probs)), chunksize=16)
            print(f"{min(i + 1024, len(paths))}/{len(paths)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "calibrate"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lm")
    ap.add_argument("--lines")
    ap.add_argument("--lst")
    ap.add_argument("--out")
    ap.add_argument("--device")
    args = ap.parse_args()
    rec = Recognizer(args.ckpt, device=args.device, lm_path=args.lm)

    if args.cmd == "score":
        paths = sorted(Path(args.lines).glob("*.png"))
        rows = score_images(rec, paths)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
            w.writeheader()
            w.writerows(rows)
        mc = np.array([r["min_conf"] for r in rows])
        for th in (0.5, 0.7, 0.8, 0.9, 0.95):
            print(f"min_conf>={th}: {int((mc >= th).sum())} / {len(rows)} lines")
    else:
        paths = [l.strip() for l in open(args.lst) if l.strip()]
        rows = score_images(rec, paths)
        gts = [Path(p[: p.rfind(".")] + ".gt.txt").read_text(encoding="utf-8") for p in paths]
        print("all lines", score([(r["text"], g) for r, g in zip(rows, gts)]))
        for th in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            for need_agree in (False, True):
                keep = [(r["text"], g) for r, g in zip(rows, gts)
                        if r["min_conf"] >= th and (r["agree"] or not need_agree)]
                if keep:
                    s = score(keep)
                    print(f"min_conf>={th:<4} agree={int(need_agree)}: keep {len(keep):3d}/{len(rows)} "
                          f"({100 * len(keep) / len(rows):4.1f}%)  CER {s['cer']:.2f}  SER {s['ser']:.1f}")


if __name__ == "__main__":
    main()
