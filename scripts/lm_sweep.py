"""Sweep LM weight (alpha), insertion bonus (beta) and LM order on dumped CTC probabilities.

    python scripts/lm_sweep.py runs/X/probs_val.pkl data/lm/cs_order5.pkl [more LMs...]
Also reports beam search *without* LM (alpha=0) as the control.
"""
import itertools
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import charlm  # noqa: E402
from metrics import score  # noqa: E402


def gt(img):
    return Path(img[: img.rfind(".")] + ".gt.txt").read_text(encoding="utf-8")


def main():
    import os
    alphas = [float(a) for a in os.environ.get("ALPHAS", "0,0.2,0.35,0.5,0.75,1.0").split(",")]
    betas = [float(b) for b in os.environ.get("BETAS", "0,0.5,1.0").split(",")]
    d = pickle.load(open(sys.argv[1], "rb"))
    labels, lines = d["labels"], d["lines"]
    gts = [gt(img) for img, _ in lines]
    base = [charlm.ctc_beam_search(p, labels, None, beam=1) for _, p in lines]
    print("greedy", json.dumps(score(zip(base, gts))))
    results = []
    for lm_path in sys.argv[2:]:
        lm = charlm.load(lm_path)
        for alpha, beta in itertools.product(alphas, betas):
            if alpha == 0 and beta:
                continue
            preds = [charlm.ctc_beam_search(p, labels, lm, alpha, beta, beam=12) for _, p in lines]
            s = score(zip(preds, gts))
            results.append((s["cer"], Path(lm_path).stem, alpha, beta, s))
            print(Path(lm_path).stem, alpha, beta, json.dumps({k: round(v, 2) for k, v in s.items()}), flush=True)
    best = min(results, key=lambda r: r[0])
    print("BEST", best)


if __name__ == "__main__":
    main()
