"""Collect liturgy texts (and, where they exist, hand-timed recordings) from the tasbeha.org hymn library.

    python liturgy/scripts/tasbeha.py lessons   # ~75 lessons with word timings + their mp3s (gold for evaluating alignment)
    python liturgy/scripts/tasbeha.py texts     # every text in the library: Coptic, English letters, English, Arabic

Polite: one request per second, robots.txt allows /hymn_library/ (it disallows /api/ and search, which we don't use).
Each hymn becomes one JSON file:
  {"id", "mid", "title", "category", "audio", "lines": [{"idx", "start", "end",
     "words": [{"coptic", "translit", "start"}], "english", "arabic"}]}
start/end are seconds in the recording (lessons only; tasbeha sets them by hand).
"""
import argparse
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://tasbeha.org"
OUT = Path(__file__).resolve().parent.parent / "data/tasbeha"
UA = "Mozilla/5.0 (X11; Linux x86_64) coptic-liturgy-study/0.1"
_last = [0.0]


def get(path, binary=False):
    wait = 1.0 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(4):
        try:
            req = urllib.request.Request(BASE + path if path.startswith("/") else path, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            _last[0] = time.time()
            return data if binary else data.decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt + 1} {path}: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed: {path}")


def text_of(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(s))).strip()


def parse_hymn(s):
    """Lines with per-word Coptic/English-letter pairs, translations and (if present) timings."""
    title = text_of(re.search(r"<title>(.*?)</title>", s, re.S).group(1)).split(" | ")[0]
    lines = []
    for m in re.finditer(r'<div class="sync-line row[^"]*" data-idx="(\d+)"(.*?)(?=<div class="sync-line row|<div class="sync-para"|</div></div>\s*</div>\s*<dl|$)',
                         s, re.S):
        body = m.group(2)
        words = []
        for wp in re.finditer(r'<span class="wp" data-w="(\d+)">(.*?)</span></span>', body, re.S):
            w = wp.group(2)
            tr = re.search(r'class="w sync-translit" data-w="\d+">(.*?)</span>', w, re.S)
            cp = re.search(r'class="w sync-coptic[^"]*" data-w="\d+"[^>]*>(.*?)(?:</span>|$)', w, re.S)
            words.append({"coptic": text_of(cp.group(1)) if cp else "", "translit": text_of(tr.group(1)) if tr else ""})
        en = re.search(r'<div class="sync-english">(.*?)</div>', body, re.S)
        ar = re.search(r'<div class="sync-arabic[^"]*"[^>]*>(.*?)</div>', body, re.S)
        lines.append({"idx": int(m.group(1)), "words": words,
                      "english": text_of(en.group(1)) if en else "", "arabic": text_of(ar.group(1)) if ar else ""})
    t = re.search(r'data-timings="([^"]*)"', s)
    if t:
        tim = {x["idx"]: x for x in json.loads(html.unescape(t.group(1)))}
        for ln in lines:
            x = tim.get(ln["idx"])
            if not x:
                continue
            ln["start"], ln["end"] = x["start"], x["end"]
            for wi, ws in x["words"]:
                if wi < len(ln["words"]):
                    ln["words"][wi]["start"] = ws
    a = re.search(r'<audio src="([^"]+)"', s)
    place = re.search(r'<dd class="hp-place"><span>(.*?)</span>', s, re.S)
    rec = re.search(r"<dt>The recording</dt><dd>(.*?)</dd>", s, re.S)
    return {"title": title, "category": text_of(place.group(1)) if place else "",
            "recording": text_of(rec.group(1)) if rec else "",
            "audio": urllib.parse.urljoin(BASE, html.unescape(a.group(1))) if a else None, "lines": lines}


def lessons():
    d = OUT / "lessons"
    d.mkdir(parents=True, exist_ok=True)
    pairs = []
    for g in ("", "?g=1", "?g=2"):
        pairs += re.findall(r'data-lesson="(\d+)-(\d+)"', get("/hymn_library/lessons" + g))
    pairs = sorted(set(pairs))
    print(len(pairs), "lessons")
    for mid, vid in pairs:
        js = d / f"{vid}_{mid}.json"
        if js.exists():
            continue
        h = parse_hymn(get(f"/hymn_library/view/{vid}?mid={mid}"))
        h.update(id=int(vid), mid=int(mid))
        if h["audio"]:
            mp3 = d / f"{vid}_{mid}.mp3"
            if not mp3.exists():
                mp3.write_bytes(get(h["audio"], binary=True))
        js.write_text(json.dumps(h, ensure_ascii=False, indent=1), encoding="utf-8")
        timed = sum(1 for ln in h["lines"] if ln.get("end"))
        print(f"{vid}_{mid} {h['title'][:50]!r} lines={len(h['lines'])} timed={timed}", flush=True)


def texts():
    d = OUT / "texts"
    d.mkdir(parents=True, exist_ok=True)
    seen_cat, todo, views = set(), ["/hymn_library/all-categories"], set()
    while todo:
        p = todo.pop()
        s = get(p)
        for c in re.findall(r"""href=['"](/hymn_library/cat/\d+)['"]""", s):
            if c not in seen_cat:
                seen_cat.add(c)
                todo.append(c)
        views |= set(re.findall(r"""href=['"]/hymn_library/view/(\d+)['"]""", s))
    print(len(seen_cat), "categories,", len(views), "texts", flush=True)
    for vid in sorted(views, key=int):
        js = d / f"{vid}.json"
        if js.exists():
            continue
        h = parse_hymn(get(f"/hymn_library/view/{vid}"))
        h.update(id=int(vid))
        js.write_text(json.dumps(h, ensure_ascii=False, indent=1), encoding="utf-8")
        print(vid, h["title"][:60], len(h["lines"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lessons", "texts"])
    {"lessons": lessons, "texts": texts}[ap.parse_args().cmd]()
