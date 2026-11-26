"""Make line-level training labels for printed Bible pages by aligning to the edition text.

For each page located in the Bible (bible_ref.py locate), align the page's
predicted letters to the reference window (semi-global DP), cut the reference at
the line boundaries, and use the edition's text for that span (with word
spaces and diacritics, converted to SCAM conventions) as the line's label.
Lines whose prediction disagrees with their aligned span by more than
--max-cer are dropped (probably misaligned or mis-segmented).

    python scripts/align_printed_gt.py --lines data/horner_v3_lines --preds DIR --refs refs.json --out data/printed_gt/v3
"""
import argparse
import glob
import html
import json
import re
import sys
import unicodedata as ud
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bible_ref import BIBLE, letters  # noqa: E402
from metrics import edit_distance  # noqa: E402
from scam_convention import to_scam  # noqa: E402


def is_letter(c):
    return "Ⲁ" <= c <= "⳿" or "Ϣ" <= c <= "ϯ"


def chapter_raw(name):
    """Edition text of an NT chapter with word spaces, SCAM conventions, plus letter->raw index map."""
    f = BIBLE / "nt" / f"{name}.tt"
    toks = [html.unescape(m) for m in re.findall(r'orig_group="([^"]*)"', f.read_text(encoding="utf-8"))]
    raw = " ".join(to_scam(t) for t in toks)
    raw = ud.normalize("NFD", raw)
    pos = [i for i, c in enumerate(raw) if is_letter(c)]
    return raw, pos


def semiglobal_path(P, R):
    """Align all of P to a substring of R. Returns list of (i, j) for each P index -> R index (or None)."""
    n, m = len(P), len(R)
    INF = 10 ** 9
    prev = [0] * (m + 1)  # free start in R
    back = []
    for i in range(1, n + 1):
        cur = [INF] * (m + 1)
        bp = bytearray(m + 1)
        cur[0] = i
        bp[0] = 1
        pc = P[i - 1]
        for j in range(1, m + 1):
            d = prev[j - 1] + (pc != R[j - 1])
            u = prev[j] + 1          # P char unmatched (insertion in prediction)
            l = cur[j - 1] + 1       # R char skipped (deletion)
            if d <= u and d <= l:
                cur[j], bp[j] = d, 0
            elif u <= l:
                cur[j], bp[j] = u, 1
            else:
                cur[j], bp[j] = l, 2
        back.append(bp)
        prev = cur
    j = min(range(m + 1), key=lambda k: prev[k])  # free end in R
    i = n
    amap = [None] * n
    while i > 0:
        b = back[i - 1][j]
        if b == 0:
            amap[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif b == 1:
            i -= 1
        else:
            j -= 1
    return amap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cer", type=float, default=0.30)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    refs = json.load(open(a.refs))
    seg = {json.loads(l)["page"]: json.loads(l) for l in open(Path(a.lines) / "segmentation.jsonl")}
    kept = tot = 0
    lst = []
    cache = {}
    for page, r in refs.items():
        if r["chapter"] not in cache:
            cache[r["chapter"]] = chapter_raw(r["chapter"])
        raw, pos = cache[r["chapter"]]
        R = letters(raw)
        pred_lines = (Path(a.preds) / f"{page}.pred.txt").read_text(encoding="utf-8").split("\n")
        files = [l["file"] for l in seg[page]["lines"]]
        pred_lines = pred_lines[: len(files)]
        L = [letters(t) for t in pred_lines]
        P = "".join(L)
        w0, w1 = r["start"], r["end"]
        amap = semiglobal_path(P, R[w0:w1])
        k = 0
        for f, lt in zip(files, L):
            tot += 1
            idx = [amap[i] for i in range(k, k + len(lt)) if amap[i] is not None]
            k += len(lt)
            if len(lt) < 5 or len(idx) < 3:
                continue
            ra, rb = w0 + min(idx), w0 + max(idx) + 1
            ref_letters = R[ra:rb]
            if edit_distance(lt, ref_letters) > a.max_cer * len(ref_letters):
                continue
            s = pos[ra]
            e = pos[rb - 1] + 1
            while e < len(raw) and ud.combining(raw[e]):
                e += 1
            label = ud.normalize("NFC", raw[s:e]).strip()
            src = (Path(a.lines) / f).resolve()
            name = f"{Path(a.lines).name}__{f}"
            dst = out / name
            if not dst.exists():
                dst.symlink_to(src)
            (out / (name[:-4] + ".gt.txt")).write_text(label, encoding="utf-8")
            lst.append(str(dst.absolute()))
            kept += 1
    (out / "train.lst").write_text("\n".join(lst) + "\n")
    print(f"kept {kept} / {tot} lines from {len(refs)} pages")


if __name__ == "__main__":
    main()
