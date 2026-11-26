"""CER / SER as defined in the SCAM paper (arXiv 2606.15987, Sec. 4).

CER: total edit distance / total ground-truth length, summed over the whole set.
     Primary unit is the *grapheme cluster* (a letter plus its combining marks,
     e.g. "ⲙ̇" counts as one), matching the paper. Code-point CER is also given.
cer_nospace: CER with all spaces removed from both sides (secondary; see EXPERIMENTS.md).
SER: total line-level mismatches / number of lines (scriptio continua, so a
     "word" error rate is not meaningful).
"""
import unicodedata

from textnorm import normalize


def graphemes(s: str) -> list[str]:
    out: list[str] = []
    for ch in s:
        if out and unicodedata.combining(ch):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def edit_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(pairs):
    """pairs: iterable of (prediction, ground_truth). Returns dict of percentages."""
    g_err = g_len = c_err = c_len = s_err = n = ns_err = ns_len = 0
    for pred, gt in pairs:
        pred, gt = normalize(pred), normalize(gt)
        pg, gg = graphemes(pred), graphemes(gt)
        g_err += edit_distance(pg, gg)
        g_len += len(gg)
        c_err += edit_distance(pred, gt)
        c_len += len(gt)
        s_err += pred != gt
        n += 1
        pn, gn = graphemes(pred.replace(" ", "")), graphemes(gt.replace(" ", ""))
        ns_err += edit_distance(pn, gn)
        ns_len += len(gn)
    return {
        "cer": 100 * g_err / max(g_len, 1),
        "cer_codepoint": 100 * c_err / max(c_len, 1),
        "ser": 100 * s_err / max(n, 1),
        # secondary: spaces removed on both sides (SCAM marks visual gaps / some pages word spaces inconsistently)
        "cer_nospace": 100 * ns_err / max(ns_len, 1),
        "lines": n,
    }
