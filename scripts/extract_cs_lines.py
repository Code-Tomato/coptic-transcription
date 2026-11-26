"""Extract diplomatic manuscript lines from Coptic Scriptorium .tt files.

Each <lb_n> element is one physical manuscript line. We join its <orig_group>
tokens without spaces (Sahidic manuscripts are written in scriptio continua),
which matches how SCAM transcribes lines.

Writes one line per row: corpus \t document \t text
Usage: extract_cs_lines.py <corpora_root> <out.tsv> [--stats]
"""
import collections
import html
import re
import sys
import unicodedata as ud
from pathlib import Path

ATTR = re.compile(r'orig_group="([^"]*)"')


def lines_from_tt(path: Path):
    cur = None
    for raw in path.open(encoding="utf-8", errors="replace"):
        if raw.startswith("<lb_n"):
            cur = []
        elif raw.startswith("</lb_n>"):
            if cur:
                yield "".join(cur)
            cur = None
        elif cur is not None and raw.startswith("<orig_group "):
            m = ATTR.search(raw)
            if m:
                cur.append(html.unescape(m.group(1)))


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    stats = "--stats" in sys.argv
    counts = collections.Counter()
    per_corpus = collections.Counter()
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for tt in sorted(root.glob("*/**/*.tt")):
            corpus = tt.relative_to(root).parts[0]
            for line in lines_from_tt(tt):
                line = ud.normalize("NFC", line.strip())
                if not line:
                    continue
                f.write(f"{corpus}\t{tt.stem}\t{line}\n")
                per_corpus[corpus] += 1
                n += 1
                if stats:
                    counts.update(ch for ch in line if not ("Ⲁ" <= ch <= "⳿" or "Ϣ" <= ch <= "ϯ"))
    print(n, "lines from", len(per_corpus), "corpora")
    for c, k in per_corpus.most_common():
        print(f"  {c:40s} {k}")
    if stats:
        print("non-letter characters:")
        for ch, k in counts.most_common(60):
            print(f"  U+{ord(ch):04X} {ud.name(ch, '?')[:45]:45s} {ch!r} {k}")


if __name__ == "__main__":
    main()
