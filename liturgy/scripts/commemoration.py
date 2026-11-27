"""Coptic Chanter's hard case: the Commemoration of the Saints, reader and choir alternating (see their README).

    python liturgy/scripts/commemoration.py align      # both recordings, our default pipeline, + per-word confidence
    python liturgy/scripts/commemoration.py choir      # choir-verse experiments, scored on the hand-timed Coptic

Data (shared by Refaat Yakoub, not in git): liturgy/data/refaat/For Nathan - Commemoration/
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from align import Aligner, emissions_cached, norm, viterbi, FRAME  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/refaat/For Nathan - Commemoration"
OUT = ROOT / "runs/commemoration"
FT = "codetomato/coptic-chant-aligner"
PIPE = dict(garbage=-1.02, skip_line=-9.0, blank_penalty=1.0, hold="vowels")


def texts():
    cop = list(csv.DictReader(open(D / "text-coptic.tsv"), delimiter="\t"))
    eng = [l.rstrip("\n").split("\t", 1) for l in open(D / "text-english-sung.txt") if l.strip()]
    gold = {}
    for r in csv.DictReader(open(D / "coptic-hand-timed-word-starts.tsv"), delimiter="\t"):
        gold[(int(r["line"][1:]), int(r["word_index"]) - 1)] = float(r["start_s"])
    lines_cop = [(int(r["line"][1:]), r["transliteration"].split()) for r in cop]
    lines_eng = [(int(k[1:]), t.split()) for k, t in eng]
    return lines_cop, lines_eng, gold


def mixed_emissions(mp3, mode="window"):
    base, ft = Aligner(), Aligner(model=FT)
    eb, _ = emissions_cached(base, mp3, mode)
    ef, _ = emissions_cached(ft, mp3, mode)
    n = min(len(eb), len(ef))
    return base, np.logaddexp(np.log(0.5) + ef[:n], np.log(0.5) + eb[:n])


def align_words(al, em, lines):
    """Our default pipeline. Returns {(line, word): (start_s, confidence)} where confidence is the lowest
    log-prob among the word's letters at the frames they were aligned to (as in Coptic Chanter's README)."""
    words, keys, breaks, toks, owner = [], [], [], [], []
    for ln, ws in lines:
        for wi, w in enumerate(ws):
            tw = al.tokens(w)
            if wi == 0 and tw:
                breaks.append(len(toks))
            for t in tw:
                toks.append(t)
                owner.append(len(keys))
            keys.append((ln, wi))
    e = em.copy()
    e[:, al.blank] -= PIPE["blank_penalty"]
    hold = {al.vocab[c] for c in "aeiou"}
    frames = viterbi(e, toks, al.blank, PIPE["garbage"], breaks, PIPE["skip_line"], hold)
    out = {}
    for k, (f0, f1) in enumerate(frames):
        if f0 is None:
            continue
        key = keys[owner[k]]
        sc = float(em[f0:f1 + 1, toks[k]].max())  # best frame of this letter
        s, c = out.get(key, (f0 * FRAME, 0.0))
        out[key] = (s, min(c, sc))
    return out


def score(pred, gold, keys=None):
    keys = [k for k in (keys or gold) if k in gold]
    e = np.array([abs(pred[k] - gold[k]) if k in pred and pred[k] is not None else 1e3 for k in keys])
    return dict(n=len(e), w05=round(100 * float((e <= 0.5).mean()), 1), w1=round(100 * float((e <= 1).mean()), 1))


def consistency(pred, lines):
    """Same tune every verse: a correct alignment puts the shared words at similar offsets in every verse.
    Returns the median spread (s) of 'on our behalf' after 'Intercede' and of 'sins' after 'that', per voice."""
    out = {}
    for voice, par in (("reader", 1), ("choir", 0)):
        sp = []
        for ln, ws in lines:
            if ln % 2 != par:
                continue
            low = [re.sub(r"\W", "", w.lower()) for w in ws]
            if low[:4] == ["intercede", "on", "our", "behalf"] and (ln, 0) in pred and (ln, 3) in pred:
                sp.append(("open", pred[(ln, 3)] - pred[(ln, 0)]))
            if "that" in low and low[-1] == "sins":
                i = len(low) - 1 - low[::-1].index("that")
                if (ln, i) in pred and (ln, len(low) - 1) in pred:
                    sp.append(("close", pred[(ln, len(low) - 1)] - pred[(ln, i)]))
        res = {}
        for kind in ("open", "close"):
            v = np.array([x for k, x in sp if k == kind])
            res[kind] = round(float(np.median(np.abs(v - np.median(v)))), 2) if len(v) else None
        out[voice] = res
    return out


def cmd_align(mode="window"):
    OUT.mkdir(parents=True, exist_ok=True)
    lc, le, gold = texts()
    res = {}
    print(f"== emissions mode: {mode}")
    for name, mp3, lines in (("coptic", D / "coptic-tasbeha.mp3", lc), ("english", D / "english-st-antony.mp3", le)):
        al, em = mixed_emissions(mp3, mode)
        a = align_words(al, em, lines)
        res[name] = {f"{k[0]}:{k[1]}": v for k, v in a.items()}
        conf = {}
        for (ln, wi), (s, c) in a.items():
            conf.setdefault(ln, []).append(c)
        med = {ln: float(np.median(v)) for ln, v in conf.items()}
        odd = [m for ln, m in med.items() if ln % 2 == 1]
        even = [m for ln, m in med.items() if ln % 2 == 0]
        print(f"{name}: {len(a)} words aligned; median word confidence odd verses {np.median(odd):.1f}, even verses {np.median(even):.1f}")
        if name == "coptic":
            p = {k: v[0] for k, v in a.items()}
            print("  vs volunteer, all:", score(p, gold),
                  " odd:", score(p, gold, [k for k in gold if k[0] % 2]),
                  " even:", score(p, gold, [k for k in gold if not k[0] % 2]))
        else:
            print("  spread of shared words across verses (median abs. deviation, s):",
                  consistency({k: v[0] for k, v in a.items()}, lines))
    (OUT / f"align_{mode}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    {"align": cmd_align}[sys.argv[1]](*sys.argv[2:])
