"""Word-level alignment of a liturgy recording to its (known) text, and evaluation against hand timings.

    python liturgy/scripts/align.py eval [--limit N]            # all tasbeha lessons -> liturgy/runs/<name>/
    python liturgy/scripts/align.py one HYMN.json AUDIO [--out X.json]

Method: CTC forced alignment. The acoustic model is MMS-300m forced-aligner (wav2vec2 trained on 1,130
languages, romanised characters a-z and '). Coptic is not one of its languages, so the text goes in as
tasbeha's English-letter transliteration (Greco-Bohairic pronunciation), which is close to what is sung.
Viterbi over the CTC graph gives each character a frame; a word starts at its first character.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"
SR = 16000
FRAME = 320 / SR  # wav2vec2 frame = 20 ms


def load_audio(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


class Aligner:
    def __init__(self, device=None, model=None):
        """model: a Hugging Face id or a local fine-tuned directory (default: the MMS forced aligner)."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model or MODEL
        self.tag = "" if self.model_path == MODEL else Path(self.model_path).name  # emissions cache subfolder
        self._model = None  # loaded on first use, so runs on cached emissions don't touch the GPU
        from huggingface_hub import hf_hub_download
        self.vocab = json.load(open(hf_hub_download(MODEL, "vocab.json")))
        self.blank = self.vocab["<blank>"]

    @property
    def model(self):
        if self._model is None:
            from transformers import Wav2Vec2ForCTC
            self._model = Wav2Vec2ForCTC.from_pretrained(self.model_path).to(self.device).eval()
            if self.device == "cuda":
                self._model.half()
        return self._model

    @torch.no_grad()
    def emissions(self, wav, win=30.0, ctx=4.0):
        """Log-probs (T, V) for the whole recording, in windows with context on both sides."""
        W, C = int(win * SR), int(ctx * SR)
        out = []
        for s in range(0, len(wav), W):
            a, b = max(0, s - C), min(len(wav), s + W + C)
            x = torch.from_numpy(wav[a:b]).to(self.device)
            x = (x - x.mean()) / (x.std() + 1e-7)
            dt = next(self.model.parameters()).dtype
            lp = self.model(x[None].to(dt)).logits[0].float().log_softmax(-1).cpu().numpy()
            f0 = int(round((s - a) / 320))
            n = int(round((min(len(wav), s + W) - s) / 320))
            out.append(lp[f0:f0 + n])
        return np.concatenate(out)

    def tokens(self, word):
        return [self.vocab[c] for c in norm(word) if c in self.vocab]

    def align(self, em, words, line_starts=(), garbage=None, skip_line=None, blank_penalty=0.0, hold=None):
        """CTC Viterbi. words: list of strings; line_starts: word indices that begin a line.
        Returns [(start_s, end_s)] per word (None if it has no letters)."""
        toks, owner, breaks = [], [], []
        ls = set(line_starts)
        for wi, w in enumerate(words):
            tw = self.tokens(w)
            if wi in ls and tw:
                breaks.append(len(toks))
            for t in tw:
                toks.append(t)
                owner.append(wi)
        if blank_penalty:  # make 'nothing new' less attractive than continuing the current letter
            em = em.copy()
            em[:, self.blank] -= blank_penalty
        hold_ids = None
        if hold:
            hold_ids = {self.vocab[c] for c in ("aeiou" if hold == "vowels" else "abcdefghijklmnopqrstuvwxyz") if c in self.vocab}
        frames = viterbi(em, toks, self.blank, garbage, breaks, skip_line, hold_ids)
        spans = [None] * len(words)
        for k, (f0, f1) in enumerate(frames):
            if f0 is None:
                continue
            wi = owner[k]
            s, e = f0 * FRAME, (f1 + 1) * FRAME
            spans[wi] = (s, e) if spans[wi] is None else (spans[wi][0], e)
        return spans


def norm(w):
    w = w.lower().replace("-", "")
    return re.sub(r"[^a-z']", "", w)


def viterbi(em, toks, blank, garbage=None, breaks=(), skip_line=None, hold=None):
    """Best CTC path for token sequence toks through emissions em (T, V). Returns (first, last) frame per token.

    garbage: if set, a 'garbage' state with this per-frame log-score is added at the start, the end and before
    every token index in `breaks` (line starts). It soaks up audio that is not in the text (another
    singer's part, a repeat, an introduction) instead of forcing words onto it.
    skip_line: if set (with garbage), a whole line may be left out (it isn't sung in this recording) at this
    log-score per letter of the line; its tokens then get no frames (None).
    hold: set of token ids that may be held (melisma): after the letter and a pause (blank) the path may come
    back to the same letter instead of moving on, so a vowel sung in bursts over 30 s stays one letter.
    """
    T = len(em)
    G = em.shape[1]  # column index of the garbage emission
    if garbage is not None:
        em = np.concatenate([em, np.full((T, 1), garbage, em.dtype)], 1)
    kind, ext, tok_of = [], [], []  # kind: 0 blank, 1 token, 2 garbage

    def add(k, e, ti=-1):
        kind.append(k)
        ext.append(e)
        tok_of.append(ti)

    br = set(breaks)
    if garbage is not None:
        add(2, G)
    add(0, blank)
    for i, t in enumerate(toks):
        if garbage is not None and i in br and i > 0:
            add(2, G)
            add(0, blank)
        add(1, t, i)
        add(0, blank)
    if garbage is not None:
        add(2, G)
    kind, ext, tok_of = np.array(kind), np.array(ext), np.array(tok_of)
    S = len(kind)
    skip = np.zeros(S, bool)
    for s in range(2, S):
        if kind[s - 1] != 0:
            continue
        if kind[s] == 1:      # token after a blank: may skip the blank unless it repeats the previous token
            skip[s] = kind[s - 2] != 1 or ext[s - 2] != ext[s]
        elif kind[s] == 2:    # garbage after a line's last token
            skip[s] = kind[s - 2] == 1
        elif kind[s] == 0 and kind[s - 1] == 0:
            pass
    for s in range(2, S):     # a line's first blank may be entered straight from the previous line's blank (no garbage used)
        if kind[s] == 0 and kind[s - 1] == 2:
            skip[s] = True
    NEG = -1e30
    dp = np.full(S, NEG)
    n0 = 3 if garbage is not None else 2
    dp[:n0] = em[0, ext[:n0]]
    back = np.zeros((T, S), np.int8)  # 0 stay, 1 from s-1, 2 from s-2, 3 jump from the previous garbage state
    neg1, neg2 = np.array([NEG]), np.array([NEG, NEG])
    gs = np.where(kind == 2)[0]
    jsrc = np.zeros(S, int)
    jok = np.zeros(S, bool)
    jcost = np.zeros(S)
    if skip_line is not None and len(gs) > 1:
        jsrc[gs[1:]] = gs[:-1]
        jok[gs[1:]] = True
        for a_, b_ in zip(gs[:-1], gs[1:]):  # leaving a line out costs skip_line per letter of that line
            jcost[b_] = skip_line * int((kind[a_:b_] == 1).sum())
    back_ok = np.zeros(S, bool)  # token state s may be re-entered from its following blank s+1
    if hold:
        for s_ in range(S - 1):
            back_ok[s_] = kind[s_] == 1 and kind[s_ + 1] == 0 and int(ext[s_]) in hold
    for t in range(1, T):
        rows = [dp, np.concatenate([neg1, dp[:-1]]), np.where(skip, np.concatenate([neg2, dp[:-2]]), NEG)]
        rows.append(np.where(jok, dp[jsrc] + jcost, NEG) if skip_line is not None else np.full(S, NEG))
        if hold:
            rows.append(np.where(back_ok, np.concatenate([dp[1:], neg1]), NEG))
        m = np.stack(rows)
        arg = m.argmax(0)
        dp = m[arg, np.arange(S)] + em[t, ext]
        back[t] = arg
    ends = range(max(0, S - n0), S)
    s = max(ends, key=lambda i: dp[i])
    L = len(toks)
    first, last = [None] * L, [None] * L
    for t in range(T - 1, -1, -1):
        if kind[s] == 1:
            k = tok_of[s]
            first[k] = t
            if last[k] is None:
                last[k] = t
        b_ = back[t, s]
        s = jsrc[s] if b_ == 3 else (s + 1 if b_ == 4 else s - int(b_))
    return list(zip(first, last))


def gold_words(h):
    """(line idx, word idx, gold start) for words tasbeha has timed."""
    out = []
    for ln in h["lines"]:
        if not ln.get("end"):
            continue
        for wi, w in enumerate(ln["words"]):
            st = w.get("start")
            if st is None or (st == 0 and not (wi == 0 and ln["start"] == 0)):
                continue
            out.append((ln["idx"], wi, st))
    return out


def emissions_cached(al, mp3):
    c = ROOT / "cache/emissions" / al.tag / (Path(mp3).stem + ".npy")
    if c.exists():
        return np.load(c).astype(np.float32), None
    wav = load_audio(mp3)
    em = al.emissions(wav)
    c.parent.mkdir(parents=True, exist_ok=True)
    np.save(c, em.astype(np.float16))
    return em, wav


def run_lesson(al, js, mp3, garbage=None, line_breaks=True, full_text=False, skip_line=None, blank_penalty=0.0,
               hold=None, em=None):
    h = json.loads(Path(js).read_text(encoding="utf-8"))
    # default: only the part of the text tasbeha marked as in this recording; --full-text: the whole text
    lines = [ln for ln in h["lines"] if ln["words"] and (full_text or ln.get("end"))]
    words = [w["translit"] for ln in lines for w in ln["words"]]
    keys = [(ln["idx"], wi) for ln in lines for wi in range(len(ln["words"]))]
    if em is None:
        em, _ = emissions_cached(al, mp3)
    starts, n = [], 0
    for ln in lines:
        starts.append(n)
        n += len(ln["words"])
    spans = al.align(em, words, starts if line_breaks else (), garbage, skip_line, blank_penalty, hold)
    pred = {k: sp for k, sp in zip(keys, spans)}
    errs, line_errs, after_hold = [], [], []
    gw = gold_words(h)
    for j, (li, wi, g) in enumerate(gw):
        sp = pred.get((li, wi))
        errs.append(sp[0] - g if sp else 1e3)
        if j > 0 and g - gw[j - 1][2] >= 8:  # the word after a note held 8 s or longer
            after_hold.append(errs[-1])
    for ln in lines:
        if not ln.get("end"):
            continue
        sp = pred.get((ln["idx"], 0))
        line_errs.append(sp[0] - ln["start"] if sp else 1e3)  # a timed line the aligner skipped counts as a miss
    return dict(id=Path(js).stem, title=h["title"], seconds=round(len(em) * FRAME, 1), words=len(words),
                word_err=errs, line_err=line_errs, after_hold_err=after_hold,
                pred=[{"line": k[0], "word": k[1], "start": round(sp[0], 2), "end": round(sp[1], 2)}
                      for k, sp in zip(keys, spans) if sp])


def summary(errs):
    a = np.abs(np.array(errs)) if errs else np.array([np.nan])
    return dict(n=int(len(errs)), median=round(float(np.median(a)), 3), mean=round(float(np.mean(a)), 3),
                within_0_25=round(100 * float(np.mean(a <= 0.25)), 1), within_0_5=round(100 * float(np.mean(a <= 0.5)), 1),
                within_1=round(100 * float(np.mean(a <= 1.0)), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "one"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--name", default="A01_mms_translit")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--garbage", type=float, help="log-score per frame of the skip-audio state (off if unset)")
    ap.add_argument("--no-line-breaks", action="store_true", help="skip-audio only at the start and end")
    ap.add_argument("--full-text", action="store_true", help="give the aligner the whole text, not just the recorded lines")
    ap.add_argument("--blank-penalty", type=float, default=0.0)
    ap.add_argument("--hold", choices=["vowels", "all"], help="let a letter be held across pauses (melisma)")
    ap.add_argument("--skip-line", type=float, help="log-score per letter for leaving a whole line out (off if unset)")
    ap.add_argument("--out")
    ap.add_argument("--model", help="fine-tuned model directory (default: base MMS aligner)")
    ap.add_argument("--ids", help="file with lesson ids to evaluate (one per line), e.g. a test split")
    a = ap.parse_args()
    al = Aligner(model=a.model)
    keep = set(Path(a.ids).read_text().split()) if a.ids else None
    if a.cmd == "one":
        r = run_lesson(al, a.args[0], a.args[1], a.garbage, not a.no_line_breaks, a.full_text, a.skip_line)
        Path(a.out or "align.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print(summary(r["word_err"]))
        return
    d = ROOT / "data/tasbeha/lessons"
    out = ROOT / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for js in sorted(d.glob("*.json"))[: a.limit]:
        mp3 = js.with_suffix(".mp3")
        if not mp3.exists() or (keep is not None and js.stem not in keep):
            continue
        t = time.time()
        r = run_lesson(al, js, mp3, a.garbage, not a.no_line_breaks, a.full_text, a.skip_line, a.blank_penalty, a.hold)
        res.append(r)
        s = summary(r["word_err"])
        print(f"{r['id']:>12} {r['seconds']:6.1f}s {r['words']:4d}w  median {s['median']:.2f}s  "
              f"<=0.5s {s['within_0_5']:5.1f}%  ({time.time() - t:.1f}s) {r['title'][:40]}", flush=True)
    allw = [e for r in res for e in r["word_err"]]
    alll = [e for r in res for e in r["line_err"]]
    per = sorted((summary(r["word_err"])["median"], r["id"]) for r in res if r["word_err"])
    report = dict(name=a.name, model=a.model, ids=a.ids, garbage=a.garbage, line_breaks=not a.no_line_breaks, full_text=a.full_text,
                  skip_line=a.skip_line, blank_penalty=a.blank_penalty, hold=a.hold,
                  after_long_hold=summary([e for r in res for e in r["after_hold_err"]]), lessons=len(res), words=summary(allw), line_starts=summary(alll),
                  median_of_lesson_medians=round(float(np.median([m for m, _ in per])), 3),
                  worst=per[-5:])
    (out / "eval.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (out / "pred.json").write_text(json.dumps(res, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
