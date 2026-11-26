"""Training data for fine-tuning the aligner's acoustic model on hand-timed chant (tasbeha lessons).

    python liturgy/scripts/ft_data.py      # -> liturgy/data/ft/{train,val,test}_ids.txt, segments.jsonl

Split by hymn (a hymn's two recordings stay together), so test hymns are never heard in training.
The test set always includes the hardest drawn-out hymns: Apekran, Sotees Amen, Shere Maria (Praxis),
Tai Shori and Yes Pateer (Great).

A training segment is a run of consecutive words from one timed line, cut at tasbeha's word starts:
audio [start of first word, start of the word after the last] with the letters of those words as the CTC
target. Runs stop at 25 s; a single word held longer becomes its own segment (up to 60 s, longer ones are
cut at 60 s). Words without letters (e.g. "(...)", where the patriarch's name is sung) and untimed words
end a run, so audio that isn't in the text never gets a target.
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from align import norm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LES = ROOT / "data/tasbeha/lessons"
OUT = ROOT / "data/ft"
HARD = ["699", "1974", "322", "675", "1338"]
MAX_RUN, MAX_WORD = 25.0, 60.0


def word_spans(h):
    """[(start, end, letters)] per timed line, in order; None marks a break (untimed / letterless word)."""
    out = []
    for ln in h["lines"]:
        if not ln.get("end"):
            continue
        ws = ln["words"]
        line = []
        for wi, w in enumerate(ws):
            st = w.get("start")
            ok = st is not None and not (st == 0 and not (wi == 0 and ln["start"] == 0))
            nxt = next((x.get("start") for x in ws[wi + 1:] if x.get("start")), None)
            end = nxt if nxt is not None else ln["end"]
            letters = norm(w["translit"])
            if not ok or not letters or "(" in w["translit"] or end <= st:
                line.append(None)
            else:
                line.append((st, end, letters))
        out.append(line)
    return out


def segments(lesson_id):
    h = json.loads((LES / f"{lesson_id}.json").read_text(encoding="utf-8"))
    segs = []
    for line in word_spans(h):
        run = []

        def flush():
            # CTC needs at least ~2 frames (40 ms) per letter; shorter spans are timing slips, not data
            if run and (min(run[-1][1], run[0][0] + MAX_WORD) - run[0][0]) >= 0.04 * len("".join(x[2] for x in run)) + 0.1:
                segs.append({"id": lesson_id, "start": run[0][0], "end": min(run[-1][1], run[0][0] + MAX_WORD),
                             "text": "".join(x[2] for x in run), "words": len(run)})
            run.clear()

        for x in line:
            if x is None:
                flush()
                continue
            if run and x[1] - run[0][0] > MAX_RUN:
                flush()
            run.append(x)
        flush()
    return segs


def main():
    ids = sorted(p.stem for p in LES.glob("*.json") if p.with_suffix(".mp3").exists())
    by = defaultdict(list)
    for i in ids:
        by[i.split("_")[0]].append(i)
    hymns = sorted(set(by) - set(HARD))
    random.Random(0).shuffle(hymns)
    test_h = HARD + hymns[:8]
    val_h = hymns[8:14]
    split = {"test": [i for hh in test_h for i in by[hh]], "val": [i for hh in val_h for i in by[hh]],
             "train": [i for hh in hymns[14:] for i in by[hh]]}
    OUT.mkdir(parents=True, exist_ok=True)
    for k, v in split.items():
        (OUT / f"{k}_ids.txt").write_text("\n".join(sorted(v)) + "\n")
    n = 0
    with open(OUT / "segments.jsonl", "w") as f:
        for k in ("train", "val"):
            for i in split[k]:
                for s in segments(i):
                    s["split"] = k
                    f.write(json.dumps(s) + "\n")
                    n += 1
    for k in ("train", "val", "test"):
        segs = [s for i in split[k] for s in segments(i)]
        dur = sum(s["end"] - s["start"] for s in segs)
        print(f"{k:5} {len(split[k]):2d} recordings, {len(segs):4d} segments, {dur / 60:5.1f} min of timed chant, "
              f"longest {max((s['end'] - s['start'] for s in segs), default=0):.0f}s")
    print(n, "segments written; test:", " ".join(split["test"]))


if __name__ == "__main__":
    main()
