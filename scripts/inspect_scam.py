"""Print SCAM line counts per split and the character inventory."""
import collections, sys, unicodedata as ud
import xml.etree.ElementTree as ET

r = ET.parse(sys.argv[1]).getroot()
lines = []
for im in r.iter("image"):
    for p in im.iter("polygon"):
        t = [a.text or "" for a in p.iter("attribute") if a.get("name") == "Content"]
        lines.append((im.get("subset"), t[0] if t else None, p.get("label")))
print(len(lines), collections.Counter(s for s, _, _ in lines), collections.Counter(l for *_, l in lines))
for s, t, _ in lines[:15]:
    print(repr(t))
c = collections.Counter(ch for _, t, _ in lines for ch in (t or ""))
print(len(c), "distinct code points")
for ch, n in c.most_common():
    print(f"U+{ord(ch):04X} {ud.name(ch, '?')[:45]:45s} {ch!r} {n}")
