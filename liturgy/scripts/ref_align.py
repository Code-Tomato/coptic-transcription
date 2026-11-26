"""Reference-guided alignment: carry word timings from a timed recording of a hymn to another recording of the
same hymn by matching melody and sound (dynamic time warping), instead of reading letters alone.

    python liturgy/scripts/ref_align.py pairs          # tasbeha hymns recorded twice: transfer A->B, score on B

Features every 100 ms:
  pitch   semitones relative to the recording's own median pitch (pYIN), so key differences don't matter
  sound   the speech model's letter probabilities (blank removed, renormalised): which vowel/consonant it hears
DTW finds the best monotonic stretch of one recording onto the other, so tempo and how long a note is held
may differ freely; ornaments and small melodic differences only add cost.
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from align import SR, load_audio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LES = ROOT / "data/tasbeha/lessons"
HOP = 0.1  # seconds per DTW frame
SUB = int(HOP / 0.02)


def pitch(mp3):
    c = ROOT / "cache/pitch" / (Path(mp3).stem + ".npy")
    if c.exists():
        return np.load(c)
    import librosa
    y = load_audio(mp3)
    f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=700, sr=SR, frame_length=1024, hop_length=320)
    f0 = np.where(voiced, f0, np.nan).astype(np.float32)
    c.parent.mkdir(parents=True, exist_ok=True)
    np.save(c, f0)
    return f0


def features(stem, em_dirs=("",)):
    """(T, 30) sound features and (T,) pitch in semitones (nan = unvoiced), pooled to HOP."""
    ems = [np.load(ROOT / "cache/emissions" / d / f"{stem}.npy").astype(np.float32) for d in em_dirs]
    em = np.logaddexp.reduce(np.stack([e[: min(len(x) for x in ems)] for e in ems]), 0) - np.log(len(ems))
    p = np.exp(em[:, 1:])  # drop blank (column 0): what letter it would be
    p /= p.sum(1, keepdims=True)
    f0 = pitch(LES / f"{stem}.mp3")
    T = min(len(p), len(f0)) // SUB
    p = p[: T * SUB].reshape(T, SUB, -1).mean(1)
    st = 12 * np.log2(f0[: T * SUB] / np.nanmedian(f0))
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore")
        st = np.nanmedian(st.reshape(T, SUB), 1)
    return p, st


def cost(fa, fb, w_sound=1.0, w_pitch=1.0):
    pa, sa = fa
    pb, sb = fb
    na = pa / np.linalg.norm(pa, axis=1, keepdims=True)
    nb = pb / np.linalg.norm(pb, axis=1, keepdims=True)
    c_sound = 1 - na @ nb.T
    va, vb = ~np.isnan(sa), ~np.isnan(sb)
    d = np.abs(np.nan_to_num(sa)[:, None] - np.nan_to_num(sb)[None, :])
    c_pitch = np.where(va[:, None] & vb[None, :], np.minimum(d, 4) / 4,
                       np.where(va[:, None] ^ vb[None, :], 0.5, 0.0))
    return (w_sound * c_sound + w_pitch * c_pitch).astype(np.float32)


def transfer(fa, fb, times_a, **w):
    """Map times in recording A to recording B along the DTW path."""
    import librosa
    C = cost(fa, fb, **w)
    _, wp = librosa.sequence.dtw(C=C, backtrack=True)
    wp = wp[::-1]
    out = []
    for t in times_a:
        i = min(int(round(t / HOP)), len(fa[0]) - 1)
        js = wp[wp[:, 0] == i, 1]
        out.append(float(js.min()) * HOP if len(js) else None)
    return out


def gold(stem):
    h = json.loads((LES / f"{stem}.json").read_text(encoding="utf-8"))
    g = {}
    for ln in h["lines"]:
        if not ln.get("end"):
            continue
        for wi, w in enumerate(ln["words"]):
            st = w.get("start")
            if st is not None and not (st == 0 and not (wi == 0 and ln["start"] == 0)):
                g[(ln["idx"], wi)] = st
    return g


def pairs():
    stems = sorted(p.stem for p in LES.glob("*.json") if p.with_suffix(".mp3").exists())
    by = {}
    for s in stems:
        by.setdefault(s.split("_")[0], []).append(s)
    prs = [v for v in by.values() if len(v) == 2]
    with mp.Pool(8) as pool:
        pool.map(pitch, [LES / f"{s}.mp3" for v in prs for s in v])
    # the text aligner's predictions on the same recordings, for comparison (runs/A11 default settings)
    txt = {r["id"]: {(p["line"], p["word"]): p["start"] for p in r["pred"]}
           for r in json.load(open(ROOT / "runs/A11_b1_g1.02/pred.json"))}
    settings = {"sound only": dict(w_sound=1, w_pitch=0), "pitch only": dict(w_sound=0, w_pitch=1),
                "sound + pitch": dict(w_sound=1, w_pitch=1)}
    rows, tot = [], {k: [] for k in ["text aligner"] + list(settings)}
    for a_, b_ in prs:
        for A, B in ((a_, b_), (b_, a_)):
            ga, gb = gold(A), gold(B)
            keys = [k for k in gb if k in ga]
            if len(keys) < 3:
                continue
            fa, fb = features(A), features(B)
            res = {"text aligner": [abs(txt[B].get(k, 1e3) - gb[k]) for k in keys]}
            for name, w in settings.items():
                pred = transfer(fa, fb, [ga[k] for k in keys], **w)
                res[name] = [abs(p - gb[k]) if p is not None else 1e3 for p, k in zip(pred, keys)]
            for k, v in res.items():
                tot[k] += v
            title = json.loads((LES / f"{B}.json").read_text(encoding="utf-8"))["title"][:38]
            rows.append((f"{A} -> {B}", len(keys), {k: round(100 * float(np.mean(np.array(v) <= 0.5)), 0) for k, v in res.items()}, title))
            print(f"{A:>11} -> {B:<11} {len(keys):3d}w  " + "  ".join(f"{k}: {v:3.0f}%" for k, v in rows[-1][2].items()) + f"  {title}", flush=True)
    print("\nall pairs, words within 0.5 s / 1 s / 2 s:")
    for k, v in tot.items():
        v = np.array(v)
        print(f"  {k:14} {100 * np.mean(v <= 0.5):5.1f}  {100 * np.mean(v <= 1):5.1f}  {100 * np.mean(v <= 2):5.1f}   (n={len(v)})")
    out = ROOT / "runs/R01_ref_pairs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.json").write_text(json.dumps({"rows": rows, "totals": {k: list(map(float, v)) for k, v in tot.items()}}))


if __name__ == "__main__":
    {"pairs": pairs}[sys.argv[1]]()
