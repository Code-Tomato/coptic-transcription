"""Mix the base and a fine-tuned acoustic model: p = w * p_finetuned + (1 - w) * p_base, then align as usual.

    python liturgy/scripts/ft_mix.py --model liturgy/models/F02_lr5e-5 --split val --weights 0 0.25 0.5 0.75 1
    python liturgy/scripts/ft_mix.py --model liturgy/models/F02_lr5e-5 --split test --weights 0.5   # once, after choosing on val

Emissions come from the per-model caches (computed if missing).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from align import Aligner, emissions_cached, run_lesson, summary  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LES = ROOT / "data/tasbeha/lessons"
PIPE = dict(garbage=-1.02, line_breaks=True, full_text=True, skip_line=-9.0, blank_penalty=1.0, hold="vowels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--weights", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1])
    a = ap.parse_args()
    base, ft = Aligner(), Aligner(model=a.model)
    ids = (ROOT / f"data/ft/{a.split}_ids.txt").read_text().split()
    ems = {i: (emissions_cached(base, LES / f"{i}.mp3")[0], emissions_cached(ft, LES / f"{i}.mp3")[0]) for i in ids}
    res = {}
    for w in a.weights:
        errs, hold, per = [], [], {}
        for i in ids:
            eb, ef = ems[i]
            n = min(len(eb), len(ef))
            em = np.logaddexp(np.log(max(w, 1e-9)) + ef[:n], np.log(max(1 - w, 1e-9)) + eb[:n])
            r = run_lesson(base, LES / f"{i}.json", LES / f"{i}.mp3", em=em, **PIPE)
            errs += r["word_err"]
            hold += r["after_hold_err"]
            per[i] = round(100 * float((np.abs(r["word_err"]) <= 0.5).mean()), 1) if r["word_err"] else None
        res[w] = {"words": summary(errs), "after_hold": summary(hold), "per_lesson": per}
        print(f"{a.split} w={w:<5} words<=0.5 {res[w]['words']['within_0_5']:5.1f} <=1 {res[w]['words']['within_1']:5.1f} "
              f"| after hold <=0.5 {res[w]['after_hold']['within_0_5']:5.1f} (n={res[w]['after_hold']['n']})", flush=True)
    out = ROOT / "runs" / f"M_{Path(a.model).name}_{a.split}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mix.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
