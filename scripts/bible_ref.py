"""Out-of-domain benchmark: held-out Vatican leaves scored against the Sahidic Bible.

Many Borg. copt. 109 leaves are biblical. For held-out fascicles (never used in
training or pseudo-labelling) we locate each page's passage in the Sahidic
NT/OT (Coptic Scriptorium) and freeze a reference window. Any model is then
scored by *infix* edit distance: the page's predicted letters against the best
matching substring of the frozen window, letters only (diacritics, punctuation
and spaces removed, since manuscript and edition conventions differ).

Spelling variants between a manuscript and the edition put a floor under the
score that is the same for every model, so it's good for comparing models, not
an absolute accuracy.

    python scripts/bible_ref.py build-text                     # -> data/text/bible/bible_letters.json
    python scripts/bible_ref.py locate --preds DIR --out data/bible_eval/refs.json   (once, frozen)
    python scripts/bible_ref.py score  --preds DIR --refs data/bible_eval/refs.json
"""
import argparse
import glob
import html
import json
import re
import sys
import unicodedata as ud
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIBLE = ROOT / "data/text/bible"
K = 8  # n-gram size for locating passages


def letters(s: str) -> str:
    s = ud.normalize("NFD", s)
    return "".join(c.lower() for c in s if "Ⲁ" <= c <= "⳿" or "Ϣ" <= c <= "ϯ")


def build_text():
    books = []
    for f in sorted(glob.glob(str(BIBLE / "nt/*.tt"))):
        toks = [html.unescape(m) for m in re.findall(r'orig_group="([^"]*)"', open(f, encoding="utf-8").read())]
        books.append((Path(f).stem, letters("".join(toks))))
    for f in sorted(glob.glob(str(BIBLE / "ot/**/*"), recursive=True)):
        if not Path(f).is_file():
            continue
        txt = [l[len("# text = "):] for l in open(f, encoding="utf-8", errors="replace") if l.startswith("# text = ")]
        if txt:
            books.append((Path(f).stem, letters("".join(txt))))
    out = BIBLE / "bible_letters.json"
    json.dump(books, open(out, "w"), ensure_ascii=False)
    print(len(books), "chapters,", sum(len(t) for _, t in books), "letters ->", out)


def infix_distance(pred: str, ref: str):
    """Edit distance between pred and the best-matching substring of ref."""
    m = len(pred)
    prev = list(range(m + 1))          # column for empty ref prefix: deleting pred chars
    best = prev[m]
    for j in range(1, len(ref) + 1):
        cur = [0] * (m + 1)            # free start anywhere in ref
        rc = ref[j - 1]
        for i in range(1, m + 1):
            cur[i] = min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (pred[i - 1] != rc))
        best = min(best, cur[m])       # free end anywhere in ref
        prev = cur
    return best


def load_preds(pred_dir):
    return {Path(f).name[: -len(".pred.txt")]: letters(open(f, encoding="utf-8").read())
            for f in sorted(glob.glob(str(Path(pred_dir) / "*.pred.txt")))}


def locate(pred_dir, out, min_hit=0.25, margin=250, k=None):
    k = k or K
    books = json.load(open(BIBLE / "bible_letters.json"))
    index = defaultdict(list)
    for bi, (_, t) in enumerate(books):
        for p in range(0, len(t) - k + 1):
            index[t[p:p + k]].append((bi, p))
    refs = {}
    for page, P in load_preds(pred_dir).items():
        if len(P) < 150:
            continue
        votes = Counter()
        grams = [P[i:i + k] for i in range(0, len(P) - k + 1)]
        for i, g in enumerate(grams):
            hits = index.get(g, [])
            if 0 < len(hits) <= 5:     # ignore very common n-grams
                for bi, p in hits:
                    votes[(bi, (p - i) // 50)] += 1
        if not votes:
            continue
        (bi, ob), n = votes.most_common(1)[0]
        hit = n / max(1, len(grams))
        if hit < min_hit:
            continue
        name, t = books[bi]
        start = max(0, ob * 50 - margin)
        end = min(len(t), ob * 50 + len(P) + 2 * margin)
        refs[page] = {"chapter": name, "start": start, "end": end, "ref": t[start:end],
                      "hit_rate": round(hit, 3), "pred_len": len(P)}
        print(f"{page}: {name} hit={hit:.2f}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(refs, open(out, "w"), ensure_ascii=False, indent=1)
    print(len(refs), "biblical pages located")


def score(pred_dir, refs_path):
    refs = json.load(open(refs_path))
    preds = load_preds(pred_dir)
    err = tot = 0
    per = {}
    for page, r in refs.items():
        P = preds.get(page, "")
        d = infix_distance(P, r["ref"])
        # normalise by the predicted length (≈ page length); a model that outputs nothing gets 100%
        n = max(len(P), r["pred_len"])
        err += min(d, n)
        tot += n
        per[page] = round(100 * min(d, n) / n, 2)
    res = {"bible_cer": round(100 * err / max(tot, 1), 2), "pages": len(refs), "per_page": per}
    print(json.dumps(res, indent=1))
    return res


_INDEX = None


def _index():
    """8-gram -> [(chapter index, position)], over all chapters in canonical order (first 4 occurrences)."""
    global _INDEX
    if _INDEX is None:
        books = json.load(open(BIBLE / "bible_letters.json"))
        idx = defaultdict(list)
        for ci, (_, t) in enumerate(books):
            for i in range(len(t) - K + 1):
                l = idx[t[i:i + K]]
                if len(l) < 4:
                    l.append((ci, i))
        _INDEX = (books, idx)
    return _INDEX


def score_segmented(pred_dir, refs_path, chunk=150, margin=300):
    """v2 of `score`: a page may switch passage (lectionaries) or run across a chapter boundary.

    The page's letters are cut into `chunk`-letter pieces and each piece is located by 8-gram votes.
    Consecutive pieces from the same passage form a run (a chapter together with its neighbours, so a
    chapter boundary doesn't start a new run). Each run is scored with infix_distance against its own
    passage window, and errors are summed over runs. Recognition errors still count in full; only the
    jumps between passages are no longer charged to the model. Normalisation is the same as `score`.
    """
    books, idx = _index()
    refs = json.load(open(refs_path))
    preds = load_preds(pred_dir)
    err = tot = 0
    per = {}
    for page, r in refs.items():
        P = preds.get(page, "")
        runs = []  # [chapter index, [pieces], [(chapter, pos) hits]]
        for s in range(0, len(P), chunk):
            seg = P[s:s + chunk]
            hits = [h for i in range(len(seg) - K + 1) for h in idx.get(seg[i:i + K], ())]
            votes = Counter(c for c, _ in hits)
            top, n_top = votes.most_common(1)[0] if votes else (None, 0)
            if runs:
                cur = runs[-1][0]
                near = sum(votes[c] for c in (cur - 1, cur, cur + 1))
                if top is None or n_top < 15 or near >= 0.5 * n_top or len(seg) < chunk // 3:
                    runs[-1][1].append(seg)
                    runs[-1][2] += [h for h in hits if abs(h[0] - cur) <= 1]
                    continue
            if top is None:  # nothing located yet: start a run on the page-level reference chapter
                top = next(i for i, (name, _) in enumerate(books) if name == r["chapter"])
            runs.append([top, [seg], [h for h in hits if abs(h[0] - top) <= 1]])
        d = 0
        for ci, pieces, hits in runs:
            lo, hi = max(0, ci - 1), min(len(books), ci + 2)
            text, off = "", {}
            for c in range(lo, hi):
                off[c] = len(text)
                text += books[c][1]
            pos = [off[c] + p for c, p in hits]
            Pr = "".join(pieces)
            a, b = (min(pos), max(pos)) if pos else (off[ci], off[ci] + len(Pr))
            d += infix_distance(Pr, text[max(0, a - margin): b + len(Pr) // 4 + margin])
        n = max(len(P), r["pred_len"])
        err += min(d, n)
        tot += n
        per[page] = {"cer": round(100 * min(d, n) / n, 2), "runs": [books[c][0] for c, _, _ in runs]}
    res = {"bible_cer_v2": round(100 * err / max(tot, 1), 2), "pages": len(refs), "per_page": per}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-text", "locate", "score", "score2"])
    ap.add_argument("--preds")
    ap.add_argument("--refs")
    ap.add_argument("--out")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--min-hit", type=float, default=0.25)
    a = ap.parse_args()
    if a.cmd == "build-text":
        build_text()
    elif a.cmd == "locate":
        locate(a.preds, a.out, a.min_hit, k=a.k)
    elif a.cmd == "score2":
        score_segmented(a.preds, a.refs)
    else:
        score(a.preds, a.refs)


if __name__ == "__main__":
    sys.exit(main())
