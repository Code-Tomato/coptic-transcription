"""Side-by-side check of automatic timings against tasbeha's hand-checked ones, as a listenable page.

    python liturgy/scripts/compare_page.py 675_8028 1974_8020 ...   # tasbeha lesson ids (liturgy/data/tasbeha/lessons)

Uses the same settings as make_lesson.py (whole text, near-free skip-audio between lines, skip-line -9/letter, held vowels).
Writes liturgy/lessons/compare_<id>.html: words are underlined green/amber/red by how far the automatic start
is from tasbeha's, and a switch highlights with either set of timings so you can hear which is right.
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from align import Aligner, emissions_cached  # noqa: E402
from make_lesson import PLAYER, is_label  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


FT = ROOT / "models/F03_lr3e-5_freeze12_fp16"  # local copy of huggingface.co/codetomato/coptic-chant-aligner
MIX = 0.5


def main(ids):
    al = Aligner()
    ft = Aligner(model=str(FT)) if FT.exists() else None
    out = ROOT / "lessons"
    out.mkdir(exist_ok=True)
    for k in ids:
        js = ROOT / f"data/tasbeha/lessons/{k}.json"
        mp3 = js.with_suffix(".mp3")
        h = json.loads(js.read_text(encoding="utf-8"))
        lines = [ln for ln in h["lines"] if ln["words"]]
        sung = [ln for ln in lines if not is_label(ln)]
        words = [w["translit"] or w["coptic"] for ln in sung for w in ln["words"]]
        starts, n = [], 0
        for ln in sung:
            starts.append(n)
            n += len(ln["words"])
        em, _ = emissions_cached(al, mp3)
        if ft is not None:  # same 50/50 mix of base and fine-tuned model as make_lesson.py
            ef, _ = emissions_cached(ft, mp3)
            n_ = min(len(em), len(ef))
            em = np.logaddexp(np.log(MIX) + ef[:n_], np.log(1 - MIX) + em[:n_])
        spans = iter(al.align(em, words, starts, -1.02, -9.0, 1.0, "vowels"))
        out_lines, errs = [], []
        for ln in lines:
            lab = is_label(ln)
            ws = []
            for wi, w in enumerate(ln["words"]):
                sp = None if lab else next(spans)
                g = w.get("start") if ln.get("end") else None
                if g == 0 and not (wi == 0 and ln.get("start") == 0):
                    g = None  # tasbeha leaves untimed words at 0
                a = round(sp[0], 2) if sp else None
                ws.append({"coptic": w["coptic"], "translit": w["translit"], "start": a,
                           "end": round(sp[1], 2) if sp else None, "gold": g})
                if g is not None:
                    errs.append(abs(a - g) if a is not None else 1e3)
            st = [w["start"] for w in ws if w["start"] is not None]
            out_lines.append({"idx": ln["idx"], "label": lab, "words": ws, "english": ln["english"],
                              "arabic": ln["arabic"], "start": st[0] if st else None})
        e = np.array(errs)
        cmp = {"n": int(len(e)), "median": round(float(np.median(e)), 2),
               **{f"within_{str(t).replace('.', '_')}": round(100 * float((e <= t).mean()), 1) for t in (0.5, 1, 2)}}
        lesson = {"title": h["title"], "text_source": f"https://tasbeha.org/hymn_library/view/{h['id']}?mid={h['mid']}",
                  "audio": {"source": f"https://tasbeha.org/hymn_library/view/{h['id']}?mid={h['mid']}",
                            "title": h.get("recording") or "tasbeha.org recording"},
                  "duration": round(len(em) * 0.02, 2), "compare": cmp, "lines": out_lines}
        page = PLAYER.read_text(encoding="utf-8").replace("/*LESSON*/null", json.dumps(lesson, ensure_ascii=False))
        page = page.replace("AUDIO_SRC", "data:audio/mpeg;base64," + base64.b64encode(mp3.read_bytes()).decode())
        (out / f"compare_{k}.html").write_text(page, encoding="utf-8")
        (out / f"compare_{k}.json").write_text(json.dumps(lesson, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{k:>11} {lesson['duration']:6.0f}s {cmp['n']:4d} words  <=0.5s {cmp['within_0_5']:5.1f}%  "
              f"<=1s {cmp['within_1']:5.1f}%  <=2s {cmp['within_2']:5.1f}%  median {cmp['median']:.2f}s  {h['title'][:45]}")


if __name__ == "__main__":
    main(sys.argv[1:])
