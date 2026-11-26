"""Build the GitHub Pages demo into docs/: a landing page and one sing-along page per lesson.

    python liturgy/scripts/build_site.py

Lessons come from liturgy/lessons/*.json (made by make_lesson.py from YouTube recordings). On the site the
player embeds the original YouTube video (no audio is re-hosted); texts are credited to tasbeha.org.
"""
import html
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
DOCS = REPO / "docs"
GITHUB = "https://github.com/Code-Tomato/coptic-transcription"

# test-set numbers for the landing page (liturgy/README.md, rounds 2-3; EXPERIMENTS.md for OCR)
ALIGN = [
    ("All 75 hand-timed tasbeha recordings", "93%", "of words start within 0.5 s of the hand-checked time (A11 settings)"),
    ("13 hymns held out from training", "78%", "same measure, on a test set chosen to include the hardest drawn-out hymns"),
    ("Words right after a note held 8 s or longer", "56%", "on the held-out hymns (51% before fine-tuning on chant)"),
]
OCR = [
    ("SCAM-A test lines (official CER)", "11.5%"),
    ("SCAM-A test lines, ignoring word spaces", "8.0%"),
    ("SCAM-A test, whole pages", "7.0%"),
    ("Unseen scribes (letters vs. the Sahidic Bible)", "11.0%"),
]

CSS = """
:root{--bg:#f6f3ed;--panel:#fffdf9;--ink:#1f1c18;--muted:#6f685e;--line:#e4ddd0;--accent:#8a2d0c;--soft:#f6e3d6}
@media (prefers-color-scheme:dark){:root{--bg:#15130f;--panel:#1e1b17;--ink:#ede7dc;--muted:#a59d90;--line:#363029;--accent:#f0a46d;--soft:#3a2617}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}
main{max-width:860px;margin:auto;padding:28px 16px 60px}
h1{font-size:28px;margin:0 0 6px;line-height:1.2}h1 .cop{font-family:'Noto Sans Coptic',serif;color:var(--accent)}
h2{font-size:20px;margin:34px 0 6px}p{margin:6px 0 12px}.muted{color:var(--muted)}
a{color:var(--accent)}
.cards{list-style:none;padding:0;margin:12px 0;display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.cards li{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:0;overflow:hidden}
.cards a{display:block;text-decoration:none;color:inherit;padding:0 0 12px}
.cards img{width:100%;aspect-ratio:16/9;object-fit:cover;display:block;background:#000}
.cards b{display:block;padding:10px 12px 0;color:var(--accent)}.cards span{display:block;padding:2px 12px 0;font-size:13px;color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:12px 0}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.stat .n{font-size:28px;font-weight:700;color:var(--accent);line-height:1.1}.stat .l{font-weight:600;font-size:14px;margin-top:4px}.stat .d{font-size:13px;color:var(--muted)}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:15px}td,th{border-bottom:1px solid var(--line);padding:7px 6px;text-align:left}td.r{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.how{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:6px 16px}
.how ol{padding-left:20px}
footer{margin-top:40px;font-size:13px;color:var(--muted);border-top:1px solid var(--line);padding-top:14px}
"""


def youtube_id(url):
    m = re.search(r"[?&]v=([\w-]{11})", url or "")
    return m.group(1) if m else None


def main():
    if DOCS.exists():
        shutil.rmtree(DOCS)
    (DOCS / "lessons").mkdir(parents=True)
    (DOCS / ".nojekyll").write_text("")
    player = (ROOT / "player/player.html").read_text(encoding="utf-8")
    player = player.replace('<audio id="au" src="AUDIO_SRC" preload="auto"></audio>', "")
    player = player.replace("<header>", '<header><a href="../" style="font-size:13px;color:var(--muted)">← All lessons</a>', 1)
    cards = []
    for f in sorted((ROOT / "lessons").glob("*.json")):
        if f.name.startswith("compare_"):
            continue
        L = json.loads(f.read_text(encoding="utf-8"))
        vid = youtube_id(L["audio"].get("source"))
        if not vid:
            continue
        L["youtube"] = vid
        name = f.stem
        page = player.replace("/*LESSON*/null", json.dumps(L, ensure_ascii=False).replace("</", "<\\/"))
        (DOCS / "lessons" / f"{name}.html").write_text(page, encoding="utf-8")
        dur = f"{int(L['duration'] // 60)}:{int(L['duration'] % 60):02d}"
        cards.append(f'<li><a href="lessons/{name}.html"><img loading="lazy" alt="" src="https://i.ytimg.com/vi/{vid}/mqdefault.jpg">'
                     f'<b>{html.escape(L["title"])}</b><span>{html.escape(L["audio"].get("channel") or "")} · {dur}</span></a></li>')
    stats = "".join(f'<div class="stat"><div class="n">{n}</div><div class="l">{html.escape(l)}</div><div class="d">{html.escape(d)}</div></div>'
                    for l, n, d in ALIGN)
    ocr = "".join(f"<tr><td>{html.escape(k)}</td><td class=r>{v}</td></tr>" for k, v in OCR)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coptic Sing-Along</title><meta name="description" content="Coptic hymns with word-by-word highlighting synced to recordings, and an OCR model for Coptic manuscripts.">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Coptic&display=swap" rel="stylesheet"><style>{CSS}</style></head>
<body><main>
<h1>Coptic sing-along <span class="cop">ⲧⲁⲓϣⲟⲩⲣⲏ</span></h1>
<p>Learn Coptic hymns line by line: each word lights up as it is sung, in Coptic, English letters, English and Arabic.
Click any word to hear it, repeat a line, or slow the recording down. Timings are made automatically from a recording and its text.</p>

<h2>Lessons</h2>
<ul class="cards">{"".join(cards)}</ul>
<p class="muted">Recordings play from YouTube; texts and translations are from the <a href="https://tasbeha.org/hymn_library/">tasbeha.org hymn library</a>.
Each lesson page has a <b>Fix timings</b> mode for teachers.</p>

<h2>How accurate is it?</h2>
<p>Checked against hand-timed recordings from tasbeha.org:</p>
<div class="stats">{stats}</div>
<p class="muted">Short responses and litanies are usually 95–100% right. The hard cases are long melismas, where one syllable is sung for 30–70 seconds,
and recordings that don't match the text (a different patriarch's name, a spoken introduction).</p>

<div class="how"><h2 style="margin-top:12px">How it works</h2><ol>
<li><b>Text:</b> the hymn from tasbeha.org, where every Coptic word comes with its English-letter pronunciation (Greco-Bohairic).</li>
<li><b>Sound:</b> Meta's MMS speech model (trained on 1,100+ languages), mixed 50/50 with <a href="https://huggingface.co/codetomato/coptic-chant-aligner">a copy fine-tuned on 108 minutes of hand-timed Coptic chant</a>, hears which letters are being sung every 20 ms.</li>
<li><b>Alignment:</b> forced alignment places the known words in order on the recording, skipping audio that isn't in the text and lines that aren't sung, and letting a vowel be held through a melisma.</li>
</ol></div>

<h2>Also: reading Coptic manuscripts</h2>
<p>The same project trained an OCR model for Sahidic Coptic manuscripts and print (line finder + ResNet-50/BiLSTM reader + Coptic language model),
evaluated on the <a href="https://huggingface.co/datasets/blowing-up-groundhogs/scam">SCAM</a> benchmark (character error rate, lower is better):</p>
<table>{ocr}</table>
<p class="muted">Published results on the same test lines: VAN 7.50, CRNN 13.91, TrOCR-L 14.99. Details in <a href="{GITHUB}/blob/main/EXPERIMENTS.md">EXPERIMENTS.md</a>.</p>

<footer>Code: <a href="{GITHUB}">{GITHUB.replace("https://", "")}</a> (MIT license). Texts © their authors, from tasbeha.org; recordings © their owners, embedded from YouTube.
Speech models: <a href="https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner">MMS-300m forced aligner</a> and its chant fine-tune <a href="https://huggingface.co/codetomato/coptic-chant-aligner">codetomato/coptic-chant-aligner</a>.</footer>
</main></body></html>"""
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    print(f"docs/: {len(cards)} lessons")


if __name__ == "__main__":
    main()
