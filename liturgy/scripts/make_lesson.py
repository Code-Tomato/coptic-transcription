"""Align any recording to a tasbeha text and write a lesson: a JSON file plus a self-contained player page.

    python liturgy/scripts/make_lesson.py --text 675 --audio youtube:4YWKluIeY8c [--lines 0-1] [--garbage -2]
    python liturgy/scripts/make_lesson.py --text 32 --audio path/to/recording.mp3

--text      tasbeha text id (liturgy/data/tasbeha/texts/<id>.json) or a path to such a JSON file
--audio     a local file, or youtube:<id> (downloaded with yt-dlp into liturgy/data/youtube/)
--lines     which lines of the text the recording covers, e.g. 3-17 (default: all)
Output: liturgy/lessons/<name>.json and liturgy/lessons/<name>.html (audio embedded, opens offline).
"""
import argparse
import base64
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from align import SR, Aligner, emissions_for, load_audio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PLAYER = ROOT / "player/player.html"
# lines like "Deacon:" / "People:" name who sings next; they are shown but not aligned
LABEL = re.compile(r"^(the )?(deacon|people|congregation|priest|reader|cantor|choir|all|bishop|pope)s?:$", re.I)


def is_label(ln):
    return len(ln["words"]) <= 2 and bool(LABEL.match(ln["english"].strip()))


def fetch_audio(spec):
    if spec.startswith("youtube:"):
        vid = spec.split(":", 1)[1]
        d = ROOT / "data/youtube"
        d.mkdir(parents=True, exist_ok=True)
        mp3 = d / f"{vid}.mp3"
        if not mp3.exists():
            subprocess.run(["yt-dlp", "-q", "-f", "bestaudio", "-x", "--audio-format", "mp3", "--audio-quality", "5",
                            "--write-info-json", "-o", str(d / "%(id)s.%(ext)s"), f"https://www.youtube.com/watch?v={vid}"],
                           check=True)
        info = json.loads((d / f"{vid}.info.json").read_text())
        return mp3, {"source": f"https://www.youtube.com/watch?v={vid}", "title": info.get("title"),
                     "channel": info.get("channel")}
    return Path(spec), {"source": Path(spec).name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--lines")
    # defaults from the A09-A11 sweeps: skipping audio between lines costs barely more than a pause
    ap.add_argument("--garbage", type=float, default=-1.02, help="per-frame score of the skip-audio state")
    ap.add_argument("--blank-penalty", type=float, default=1.0)
    ap.add_argument("--hold", default="vowels", help="vowels | all | none: letters that may be held (melisma)")
    ap.add_argument("--skip-line", type=float, default=-9.0, help="per-letter score for leaving out a line not sung")
    ap.add_argument("--name")
    ap.add_argument("--ft-model", default="codetomato/coptic-chant-aligner",
                    help="acoustic model fine-tuned on hand-timed chant (Hugging Face id or local folder), "
                         "mixed with the base model ('' to turn off)")
    ap.add_argument("--mix", type=float, default=0.5, help="weight of the fine-tuned model's probabilities")
    ap.add_argument("--emissions", default="auto", choices=["auto", "window", "pauses"],
                    help="auto: 30 s windows, falling back to pause-cut pieces where a voice gets drowned out")
    a = ap.parse_args()

    tp = Path(a.text) if a.text.endswith(".json") else ROOT / f"data/tasbeha/texts/{a.text}.json"
    h = json.loads(tp.read_text(encoding="utf-8"))
    lines = [ln for ln in h["lines"] if ln["words"]]
    if a.lines:
        lo, hi = (int(x) for x in a.lines.split("-"))
        lines = [ln for ln in lines if lo <= ln["idx"] <= hi]
    mp3, src = fetch_audio(a.audio)

    al = Aligner()
    wav = load_audio(mp3)
    em = emissions_for(al, wav, a.emissions)
    mixed = bool(a.ft_model) and a.mix > 0
    if mixed:  # p = mix * p_finetuned + (1 - mix) * p_base  (test set: better on held notes, same elsewhere)
        import numpy as np
        ef = emissions_for(Aligner(model=a.ft_model), wav, a.emissions)
        n = min(len(em), len(ef))
        em = np.logaddexp(np.log(a.mix) + ef[:n], np.log(1 - a.mix) + em[:n]) if a.mix < 1 else ef
    sung = [ln for ln in lines if not is_label(ln)]
    words = [w["translit"] or w["coptic"] for ln in sung for w in ln["words"]]
    starts, n = [], 0
    for ln in sung:
        starts.append(n)
        n += len(ln["words"])
    spans = iter(al.align(em, words, starts, a.garbage, a.skip_line, a.blank_penalty,
                          None if a.hold == "none" else a.hold))
    out_lines = []
    for ln in lines:
        if is_label(ln):
            out_lines.append({"idx": ln["idx"], "label": True, "words": [{"coptic": w["coptic"], "translit": w["translit"],
                              "start": None, "end": None} for w in ln["words"]], "english": ln["english"],
                              "arabic": ln["arabic"], "start": None, "end": None})
            continue
        ws = []
        for w in ln["words"]:
            sp = next(spans)
            ws.append({"coptic": w["coptic"], "translit": w["translit"],
                       "start": round(sp[0], 2) if sp else None, "end": round(sp[1], 2) if sp else None})
        timed = [w for w in ws if w["start"] is not None]
        out_lines.append({"idx": ln["idx"], "words": ws, "english": ln["english"], "arabic": ln["arabic"],
                          "start": timed[0]["start"] if timed else None, "end": timed[-1]["end"] if timed else None})

    name = a.name or f"{h['id']}_{Path(mp3).stem}"
    d = ROOT / "lessons"
    d.mkdir(parents=True, exist_ok=True)
    lesson = {"title": h["title"], "text_source": f"https://tasbeha.org/hymn_library/view/{h['id']}",
              "audio": src, "duration": round(len(wav) / SR, 2), "aligner": "mms-300m forced alignment on tasbeha transliteration (garbage -1.02, blank penalty 1, skip-line -9, hold vowels)"
              + (f" + {a.mix:g} x {Path(a.ft_model).name}" if mixed else ""),
              "lines": out_lines}
    (d / f"{name}.json").write_text(json.dumps(lesson, ensure_ascii=False, indent=1), encoding="utf-8")
    audio_b64 = base64.b64encode(Path(mp3).read_bytes()).decode()
    page = PLAYER.read_text(encoding="utf-8")
    page = page.replace("/*LESSON*/null", json.dumps(lesson, ensure_ascii=False)).replace(
        "AUDIO_SRC", "data:audio/mpeg;base64," + audio_b64)
    (d / f"{name}.html").write_text(page, encoding="utf-8")
    print(d / f"{name}.html")


if __name__ == "__main__":
    main()
