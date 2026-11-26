"""Select confident pseudo-labels and materialise them as a training list.

    python scripts/select_pseudo.py data/pseudo/round1_E04.tsv data/pseudo/round1 --min-conf 0.8
Creates <out>/<file>.png (symlink) + <out>/<file>.gt.txt and <out>/train.lst.
"""
import argparse
import csv
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("tsv")
ap.add_argument("out")
ap.add_argument("--min-conf", type=float, default=0.8)
ap.add_argument("--min-len", type=int, default=4)
ap.add_argument("--no-agree", action="store_true")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(a.tsv, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"))
keep = [r for r in rows if float(r["min_conf"]) >= a.min_conf and int(r["len"]) >= a.min_len
        and (a.no_agree or r["agree"] == "1")]
lst = []
for r in keep:
    src = Path(r["file"]).resolve()
    dst = out / src.name
    if not dst.exists():
        os.symlink(src, dst)
    (out / (src.name[:-4] + ".gt.txt")).write_text(r["text"], encoding="utf-8")
    lst.append(str(dst.absolute()))
(out / "train.lst").write_text("\n".join(lst) + "\n")
pages = {Path(x).name.split("__")[0] for x in lst}
print(f"kept {len(keep)} / {len(rows)} lines from {len(pages)} pages")
