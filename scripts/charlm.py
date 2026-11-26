"""Character n-gram language model (interpolated Witten-Bell) + CTC prefix beam search.

The LM is trained on Coptic Scriptorium manuscript lines converted to SCAM
conventions (see scam_convention.py). We checked that SCAM-A val/test text is
not contained in Coptic Scriptorium (see EXPERIMENTS.md, "leakage check").

    python scripts/charlm.py train data/text/cs_lines.tsv data/lm/cs_order7.pkl --order 7
"""
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

BOS = "\x02"
EOS = "\x03"
# Marks the LM cannot predict from text: visual gaps (space) and the SCAM
# "illegible letter" dot. They get a fixed log-prob and are skipped in history.
TRANSPARENT = {" ", "."}
TRANSPARENT_LOGP = math.log(0.02)


def strip_transparent(s):
    return "".join(c for c in s if c not in TRANSPARENT)


class CharLM:
    def __init__(self, order=7):
        self.order = order
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]
        self.vocab = set()
        self._cache = {}

    def fit(self, lines):
        for line in lines:
            s = BOS * (self.order - 1) + line + EOS
            self.vocab.update(line)
            for i in range(self.order - 1, len(s)):
                c = s[i]
                for n in range(self.order):
                    h = s[i - n:i]
                    self.counts[n][h][c] += 1
        self.vocab.add(EOS)
        # freeze to plain dicts with totals
        self.tables = []
        for n in range(self.order):
            t = {}
            for h, d in self.counts[n].items():
                t[h] = (dict(d), sum(d.values()), len(d))
            self.tables.append(t)
        del self.counts
        return self

    def prob(self, hist, c):
        hist = (BOS * (self.order - 1) + hist)[-(self.order - 1):] if self.order > 1 else ""
        key = (hist, c)
        p = self._cache.get(key)
        if p is not None:
            return p
        p = 1.0 / (len(self.vocab) + 1)
        for n in range(self.order):
            h = hist[len(hist) - n:] if n else ""
            entry = self.tables[n].get(h)
            if entry is None:
                break
            d, tot, types = entry
            p = (d.get(c, 0) + types * p) / (tot + types)
        self._cache[key] = p
        return p

    def logp(self, hist, c):
        if c in TRANSPARENT:
            return TRANSPARENT_LOGP
        return math.log(self.prob(strip_transparent(hist), c))

    def perplexity(self, lines, eos=True):
        lp, n = 0.0, 0
        for line in lines:
            line = strip_transparent(line)
            for i, c in enumerate(line + (EOS if eos else "")):
                lp += self.logp(line[:i], c)
                n += 1
        return math.exp(-lp / n)

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_cache"] = {}
        return d


def load(path):
    lm = CharLM.__new__(CharLM)
    with open(path, "rb") as f:
        lm.__dict__.update(pickle.load(f))
    lm._cache = {}
    return lm


def ctc_beam_search(probs, labels, lm=None, alpha=0.5, beta=0.0, beam=16, prune=1e-3, blank=0, eos=False):
    """probs: (C, T) numpy array of per-frame label probabilities (softmaxed).
    labels: dict label_index -> string.  Returns best string."""
    C, T = probs.shape
    logp = [[math.log(max(float(v), 1e-30)) for v in probs[:, t]] for t in range(T)]
    NEG = -1e30

    def lse(a, b):
        if a < b:
            a, b = b, a
        return a if b <= NEG else a + math.log1p(math.exp(b - a))

    # beam entries: prefix -> (p_blank, p_nonblank, lm_score)
    beams = {"": (0.0, NEG, 0.0)}
    for t in range(T):
        lp_t = logp[t]
        cands = [k for k in range(C) if k != blank and probs[k, t] > prune]
        nxt = defaultdict(lambda: [NEG, NEG, 0.0])
        for prefix, (pb, pnb, lms) in beams.items():
            total = lse(pb, pnb)
            # extend with blank
            e = nxt[prefix]
            e[0] = lse(e[0], total + lp_t[blank]); e[2] = lms
            last = prefix[-1] if prefix else None
            for k in cands:
                ch = labels.get(k, "")
                if not ch:
                    continue
                newp = prefix + ch
                lm_inc = alpha * lm.logp(prefix, ch) + beta if lm is not None else 0.0
                if ch == last or (last is not None and prefix.endswith(ch)):
                    # repeated label: only from blank-ending path creates a new char
                    e2 = nxt[newp]
                    e2[1] = lse(e2[1], pb + lp_t[k]); e2[2] = lms + lm_inc
                    e = nxt[prefix]
                    e[1] = lse(e[1], pnb + lp_t[k])
                else:
                    e2 = nxt[newp]
                    e2[1] = lse(e2[1], total + lp_t[k]); e2[2] = lms + lm_inc
        beams = dict(sorted(nxt.items(), key=lambda kv: lse(kv[1][0], kv[1][1]) + kv[1][2], reverse=True)[:beam])
    # add end-of-sentence LM score
    def final(kv):
        pfx, (pb, pnb, lms) = kv
        s = lse(pb, pnb) + lms
        if lm is not None and eos:
            s += alpha * lm.logp(pfx, EOS)
        return s
    return max(beams.items(), key=final)[0]


def main():
    cmd = sys.argv[1]
    if cmd == "train":
        from scam_convention import is_allowed, to_scam
        src, out = sys.argv[2], sys.argv[3]
        order = int(sys.argv[sys.argv.index("--order") + 1]) if "--order" in sys.argv else 7
        lines = []
        for row in open(src, encoding="utf-8"):
            t = to_scam(row.rstrip("\n").split("\t")[-1])
            if is_allowed(t):
                lines.append(t)
        # hold out 2% for a perplexity check
        dev = lines[::50]
        train = [l for i, l in enumerate(lines) if i % 50]
        lm = CharLM(order).fit(train)
        print(f"order {order}: {len(train)} train lines, dev perplexity {lm.perplexity(dev):.2f}")
        lm = CharLM(order).fit(lines)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(lm.__getstate__(), f)


if __name__ == "__main__":
    main()
