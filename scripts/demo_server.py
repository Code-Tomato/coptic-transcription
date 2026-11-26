"""Small web demo: upload a photo or scan of a Coptic page (or pick an example) and see the transcription.

    .venv/bin/python scripts/demo_server.py [--host 100.77.252.12] [--port 8765]

Uses the same pipeline as transcribe.py (segmenter -> line crops -> recognizer -> LM beam search).
Standard library HTTP server only; meant for temporary demos on the tailnet, not for the internet.
"""
import argparse
import contextlib
import io
import json
import re
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
import bible_ref  # noqa: E402
from metrics import score  # noqa: E402
from recognize import Recognizer  # noqa: E402
from transcribe import DEFAULTS, load_segmenter, segment  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
FONT = D / "fonts/use/Antinoou.ttf"
MAX_UPLOAD = 30 * 1024 * 1024

EXAMPLES = [
    dict(id="scam1", kind="gold", title="Manuscript, SCAM test page",
         note="White Monastery manuscript (Borg. copt. 109, fasc. 120). Held-out test page with a gold transcription.",
         image=D / "scam_a_page/test/Borg.copt.109.cass.XXV.fasc.120_0006_fa_0001v.jpg"),
    dict(id="scam2", kind="gold", title="Manuscript, SCAM test page 2",
         note="Fascicle 159, also from the held-out test set, with a gold transcription.",
         image=D / "scam_a_page/test/Borg.copt.109.cass.XXIX.fasc.159_0004_fa_0001v.jpg"),
    dict(id="tim", kind="bible", title="Unseen scribe: 1 Timothy 1",
         note="Vatican leaf from a fascicle never used in training. Scored against the Sahidic edition.",
         image=D / "raw/vatlib/heldout/pages/Borg.copt.109.cass.XXII.fasc.86_11r.jpg",
         page="Borg.copt.109.cass.XXII.fasc.86_11r"),
    dict(id="john", kind="bible", title="Unseen scribe: John 9",
         note="Another held-out scribe (fasc. 50). Scored against the Sahidic edition.",
         image=D / "raw/vatlib/heldout/pages/Borg.copt.109.cass.XV.fasc.50_8r.jpg",
         page="Borg.copt.109.cass.XV.fasc.50_8r"),
    dict(id="lect", kind="bible", title="Unseen scribe: lectionary (Job, Jeremiah)",
         note="Held-out lectionary leaf (fasc. 32) whose readings jump from Job to Jeremiah mid-page.",
         image=D / "raw/vatlib/heldout/pages/Borg.copt.109.cass.X.fasc.32_5v.jpg",
         page="Borg.copt.109.cass.X.fasc.32_5v"),
    dict(id="luke", kind="printed", title="Printed: Horner, Luke 2",
         note="Page from Horner's printed Sahidic New Testament (1911). Scored against the edition text.",
         image=D / "raw/horner/v2_main/n41.jpg", page="n41"),
]
EX = {e["id"]: e for e in EXAMPLES}


def page_gt(xml_path):
    return re.findall(r"<TextEquiv><Unicode>(.*?)</Unicode></TextEquiv>", xml_path.read_text(encoding="utf-8"))


class Engine:
    def __init__(self, device):
        self.lock = threading.Lock()
        self.rec = Recognizer(str(DEFAULTS["rec"]), device=device, lm_path=str(DEFAULTS["lm"]))
        self.seg, self.cfg = load_segmenter(DEFAULTS["seg"], device)
        self.previews = OrderedDict()  # id -> jpeg bytes
        self.cache = {}

    def preview(self, im):
        im = im.copy()
        im.thumbnail((1800, 1800))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        pid = uuid.uuid4().hex
        self.previews[pid] = buf.getvalue()
        while len(self.previews) > 64:
            self.previews.popitem(last=False)
        return pid

    def run(self, im):
        im = ImageOps.exif_transpose(im).convert("RGB")
        t0 = time.time()
        with self.lock:
            boxes = segment(self.seg, self.cfg, im)
            crops = [im.crop(b) for b in boxes]
            probs = self.rec.probs(crops) if crops else []
            greedy = [self.rec.codec.decode_greedy(p.argmax(0).tolist()) for p in probs]
        lm = [self.rec.decode(p) for p in probs]
        return dict(width=im.width, height=im.height, boxes=boxes, lines=lm, greedy=greedy,
                    seconds=round(time.time() - t0, 1), preview=self.preview(im))

    def example(self, eid):
        if eid in self.cache:
            r = dict(self.cache[eid])
            if r["preview"] not in self.previews:
                r["preview"] = self.preview(Image.open(EX[eid]["image"]))
                self.cache[eid] = r
            return r
        e = EX[eid]
        r = self.run(Image.open(e["image"]))
        r.update(title=e["title"], note=e["note"], kind=e["kind"])
        if e["kind"] == "gold":
            gt = page_gt(e["image"].with_suffix(".xml"))
            s = score([(" ".join(r["lines"]), " ".join(gt))])
            sg = score([(" ".join(r["greedy"]), " ".join(gt))])
            r["reference"] = dict(label="Gold transcription (SCAM)", lines=gt)
            r["score"] = dict(value=s["cer_nospace"], greedy=sg["cer_nospace"],
                              label="page CER vs. gold, spaces ignored")
        else:
            refs = json.load(open(D / ("bible_eval" if e["kind"] == "bible" else "printed_eval") / "refs.json"))
            ref = refs[e["page"]]
            out, chaps = {}, [ref["chapter"]]
            for k in ("lines", "greedy"):
                if e["kind"] == "bible":  # same per-passage scoring (v2) as the OOD benchmark
                    with tempfile.TemporaryDirectory() as td:
                        (Path(td) / f"{e['page']}.pred.txt").write_text("\n".join(r[k]), encoding="utf-8")
                        (Path(td) / "refs.json").write_text(json.dumps({e["page"]: ref}))
                        with contextlib.redirect_stdout(io.StringIO()):
                            pp = bible_ref.score_segmented(td, str(Path(td) / "refs.json"))["per_page"][e["page"]]
                    out[k], chaps = pp["cer"], pp["runs"]
                else:
                    P = bible_ref.letters("".join(r[k]))
                    n = max(len(P), ref["pred_len"])
                    out[k] = round(100 * min(bible_ref.infix_distance(P, ref["ref"]), n) / n, 2)
            names = ", ".join(c.split("_", 1)[1].replace("_", " ") for c in chaps)
            r["reference"] = dict(label=f"Edition text near the page: {names} (letters only)",
                                  text=ref["ref"] if len(chaps) == 1 else
                                  "This page switches passage; the edition window below is for the first passage only.\n\n" + ref["ref"])
            r["score"] = dict(value=out["lines"], greedy=out["greedy"],
                              label="letter CER vs. the printed edition (spelling variants count as errors)")
        self.cache[eid] = r
        return r


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coptic OCR Demo</title>
<style>
@font-face{font-family:Antinoou;src:url(/font.ttf)}
:root{--bg:#f7f5f0;--panel:#fff;--ink:#1d1b18;--muted:#6b665d;--line:#e3ded3;--accent:#9a3412;--hl:rgba(234,88,12,.25);--box:rgba(154,52,18,.55)}
@media (prefers-color-scheme:dark){:root{--bg:#161412;--panel:#201d1a;--ink:#ece7df;--muted:#a39d92;--line:#3a352f;--accent:#fb923c;--hl:rgba(251,146,60,.28);--box:rgba(251,146,60,.6)}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
header{padding:20px 16px 8px;max-width:1400px;margin:auto}h1{margin:0;font-size:22px}header p{margin:4px 0 0;color:var(--muted)}
main{max-width:1400px;margin:auto;padding:8px 16px 40px}
.ex{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin:12px 0 18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:0;cursor:pointer;text-align:left;color:inherit;font:inherit;overflow:hidden}
.card:hover,.card.on{border-color:var(--accent)}.card img{width:100%;height:120px;object-fit:cover;display:block;background:#ccc}
.card span{display:block;padding:6px 8px;font-size:13px}
.up{display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:var(--panel);border:1px dashed var(--line);border-radius:10px;padding:12px}
.up label.btn{background:var(--accent);color:#fff;padding:7px 14px;border-radius:7px;cursor:pointer}
#status{color:var(--muted)}
.res{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;margin-top:16px}
@media (max-width:860px){.res{grid-template-columns:1fr}}
.pane{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;min-width:0}
svg{width:100%;height:auto;display:block}svg rect{fill:transparent;stroke:var(--box);stroke-width:2;vector-effect:non-scaling-stroke}
svg rect.hl{fill:var(--hl);stroke:var(--accent);stroke-width:3}
.meta{display:flex;gap:14px;flex-wrap:wrap;align-items:baseline;margin-bottom:8px}.meta b{font-size:26px;color:var(--accent)}
.muted{color:var(--muted);font-size:13px}
ol{margin:0;padding-left:2.6em;font-family:Antinoou,'Noto Sans Coptic',serif;font-size:19px}
ol li{padding:1px 4px;border-radius:4px;cursor:default;overflow-wrap:anywhere}ol li.hl{background:var(--hl)}
ol li::marker{font:12px system-ui;color:var(--muted)}
.ref{font-family:Antinoou,'Noto Sans Coptic',serif;font-size:17px;overflow-wrap:anywhere;white-space:pre-wrap;max-height:320px;overflow:auto;margin-top:6px}
.tog{font-size:13px;display:flex;gap:6px;align-items:center}
details{margin-top:12px}summary{cursor:pointer;color:var(--muted)}
</style></head><body>
<header><h1>Coptic OCR demo</h1>
<p>Line segmentation, a ResNet-50 + BiLSTM recognizer and a Coptic character language model, trained on SCAM plus self-labelled Vatican leaves.</p></header>
<main>
<div class="muted">Examples (none of these pages were used in training):</div>
<div class="ex" id="ex"></div>
<div class="up"><label class="btn">Upload a photo or scan<input id="file" type="file" accept="image/*" hidden></label>
<span class="muted">JPEG/PNG of a full page. Works best on manuscript pages in a single column or two.</span>
<span id="status"></span></div>
<div id="out"></div>
</main>
<script>
const EX=__EXAMPLES__;
const $=s=>document.querySelector(s);
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
$('#ex').innerHTML=EX.map(e=>`<button class="card" data-id="${e.id}"><img loading="lazy" src="/thumb/${e.id}" alt=""><span>${esc(e.title)}</span></button>`).join('');
document.querySelectorAll('.card').forEach(b=>b.onclick=()=>{document.querySelectorAll('.card').forEach(c=>c.classList.toggle('on',c===b));go(fetch('/example/'+b.dataset.id))});
$('#file').onchange=e=>{const f=e.target.files[0];if(!f)return;document.querySelectorAll('.card').forEach(c=>c.classList.remove('on'));go(fetch('/transcribe',{method:'POST',body:f}))};
let R=null;
async function go(p){$('#status').textContent='Transcribing…';$('#out').innerHTML='';
 try{const r=await p;if(!r.ok)throw new Error(await r.text());R=await r.json();$('#status').textContent='';show()}catch(e){$('#status').textContent='Error: '+e.message}}
function show(){const r=R,lm=$('#lmt')?$('#lmt').checked:true,L=lm?r.lines:r.greedy;
 const rects=r.boxes.map((b,i)=>`<rect data-i="${i}" x="${b[0]}" y="${b[1]}" width="${b[2]-b[0]}" height="${b[3]-b[1]}"/>`).join('');
 let sc='';if(r.score)sc=`<div class="meta"><b>${(lm?r.score.value:r.score.greedy).toFixed(1)}%</b><span class="muted">${esc(r.score.label)}</span></div>`;
 let ref='';if(r.reference)ref=`<details><summary>${esc(r.reference.label)}</summary><div class="ref">${esc(r.reference.lines?r.reference.lines.join('\n'):r.reference.text)}</div></details>`;
 $('#out').innerHTML=`<div class="res"><div class="pane"><svg viewBox="0 0 ${r.width} ${r.height}"><image href="/preview/${r.preview}" width="${r.width}" height="${r.height}"/>${rects}</svg></div>
 <div class="pane">${r.title?`<div><b>${esc(r.title)}</b></div><div class="muted" style="margin-bottom:8px">${esc(r.note)}</div>`:''}${sc}
 <div class="meta"><span class="muted">${r.boxes.length} lines · ${r.seconds}s</span><label class="tog"><input type="checkbox" id="lmt" ${lm?'checked':''}> language model</label></div>
 <ol>${L.map((t,i)=>`<li data-i="${i}">${esc(t)||'&nbsp;'}</li>`).join('')}</ol>${ref}</div></div>`;
 $('#lmt').onchange=show;
 const hl=i=>document.querySelectorAll('[data-i]').forEach(n=>n.classList.toggle('hl',n.dataset.i===i));
 document.querySelectorAll('[data-i]').forEach(n=>{n.onmouseenter=()=>hl(n.dataset.i);n.onmouseleave=()=>hl(null);n.onclick=()=>hl(n.dataset.i)});}
</script></body></html>"""


def make_handler(engine):
    thumbs = {}

    class H(BaseHTTPRequestHandler):
        def send(self, code, body, ctype, cache=False):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj):
            self.send(200, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

        def do_GET(self):
            p = self.path.split("?")[0]
            try:
                if p == "/":
                    ex = [dict(id=e["id"], title=e["title"]) for e in EXAMPLES]
                    self.send(200, PAGE.replace("__EXAMPLES__", json.dumps(ex)), "text/html; charset=utf-8")
                elif p == "/font.ttf":
                    self.send(200, FONT.read_bytes(), "font/ttf", cache=True)
                elif p.startswith("/thumb/") and p[7:] in EX:
                    eid = p[7:]
                    if eid not in thumbs:
                        im = Image.open(EX[eid]["image"]).convert("RGB")
                        im.thumbnail((400, 400))
                        buf = io.BytesIO()
                        im.save(buf, "JPEG", quality=80)
                        thumbs[eid] = buf.getvalue()
                    self.send(200, thumbs[eid], "image/jpeg", cache=True)
                elif p.startswith("/preview/") and p[9:] in engine.previews:
                    self.send(200, engine.previews[p[9:]], "image/jpeg", cache=True)
                elif p.startswith("/example/") and p[9:] in EX:
                    self.json(engine.example(p[9:]))
                else:
                    self.send(404, "not found", "text/plain")
            except Exception as e:  # noqa: BLE001
                self.send(500, f"{type(e).__name__}: {e}", "text/plain")

        def do_POST(self):
            if self.path != "/transcribe":
                return self.send(404, "not found", "text/plain")
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= MAX_UPLOAD:
                return self.send(413, "image must be under 30 MB", "text/plain")
            try:
                im = Image.open(io.BytesIO(self.rfile.read(n)))
                im.load()
            except Exception:  # noqa: BLE001
                return self.send(400, "could not read that file as an image", "text/plain")
            try:
                self.json(engine.run(im))
            except Exception as e:  # noqa: BLE001
                self.send(500, f"{type(e).__name__}: {e}", "text/plain")

        def log_message(self, fmt, *args):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="100.77.252.12")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    engine = Engine(a.device)

    def warm():  # run the examples once so clicks during the demo are instant
        for e in EXAMPLES:
            t = time.time()
            engine.example(e["id"])
            print(f"warmed {e['id']} in {time.time() - t:.1f}s", flush=True)

    threading.Thread(target=warm, daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(engine))
    print(f"serving on http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
