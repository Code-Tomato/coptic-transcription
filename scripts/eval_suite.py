"""Score one recognizer on all benchmarks at once and write runs/<name>/eval_suite.json.

    python scripts/eval_suite.py runs/E13_x/best.pt [--lm data/lm/cs_order5.pkl] [--workers 16] [--test]

- val:     SCAM-A val lines (CER official + spaces ignored), greedy and with LM
- ood:     35 held-out biblical leaves by unseen scribes (letters CER vs. edition, bible_ref.score)
- printed: 14 Horner Luke pages (same metric)
- test:    SCAM-A test lines, only with --test (score test only for a model we'd stand behind)
Line crops come from the frozen segmentations, so models are compared on identical inputs.
LM beam search runs in a fork pool; probabilities are computed once on the GPU.
"""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import bible_ref  # noqa: E402
from metrics import score  # noqa: E402
from recognize import Recognizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REC = None


def _decode(p):
    return REC.decode(p)


def load_image(p):
    """Decode now and close the file (thousands of lazily opened PNGs exceed the open-file limit)."""
    with Image.open(p) as im:
        return im.convert("L")


def read_lines(lst):
    paths = [l.strip() for l in open(lst) if l.strip()]
    gts = [Path(p[: p.rfind(".")] + ".gt.txt").read_text(encoding="utf-8") for p in paths]
    return paths, gts


def page_lines(lines_dir, refs_path):
    refs = json.load(open(refs_path))
    out = []
    for l in open(Path(lines_dir) / "segmentation.jsonl"):
        r = json.loads(l)
        if r["page"] in refs:
            out.append((r["page"], [str(Path(lines_dir) / x["file"]) for x in r["lines"]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--lm", default=str(ROOT / "data/lm/cs_order5.pkl"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    global REC
    REC = Recognizer(a.ckpt, lm_path=a.lm)
    run_dir = Path(a.out) if a.out else Path(a.ckpt).parent
    res = {"ckpt": a.ckpt, "alpha": REC.alpha, "beta": REC.beta}

    jobs = {"val": read_lines(ROOT / "data/scam_a/val.lst")}
    if a.test:
        jobs["test"] = read_lines(ROOT / "data/scam_a/test.lst")
    pages = {"ood": (page_lines(ROOT / "data/heldout_lines", ROOT / "data/bible_eval/refs.json"),
                     ROOT / "data/bible_eval/refs.json"),
             "printed": (page_lines(ROOT / "data/horner_lines_S01", ROOT / "data/printed_eval/refs.json"),
                         ROOT / "data/printed_eval/refs.json")}

    # one flat list of images -> probabilities -> LM decode in parallel
    flat = []
    for k, (paths, _) in jobs.items():
        flat += paths
    for k, (pl, _) in pages.items():
        for _, files in pl:
            flat += files
    probs = REC.probs([load_image(p) for p in flat])
    greedy = [REC.codec.decode_greedy(p.argmax(0).tolist()) for p in probs]
    with mp.get_context("fork").Pool(a.workers) as pool:
        lm = pool.map(_decode, probs, chunksize=8)

    i = 0
    for k, (paths, gts) in jobs.items():
        n = len(paths)
        g = score(zip(greedy[i:i + n], gts))
        s = score(zip(lm[i:i + n], gts))
        res[k] = {"cer": round(s["cer"], 2), "cer_nospace": round(s["cer_nospace"], 2), "ser": round(s["ser"], 2),
                  "greedy_cer": round(g["cer"], 2), "lines": n}
        i += n
    for k, (pl, refs) in pages.items():
        d = run_dir / "eval" / k
        d.mkdir(parents=True, exist_ok=True)
        for page, files in pl:
            (d / f"{page}.pred.txt").write_text("\n".join(lm[i:i + len(files)]) + "\n", encoding="utf-8")
            i += len(files)
        with contextlib.redirect_stdout(io.StringIO()):
            r = bible_ref.score(str(d), str(refs))
        res[k] = {"bible_cer": r["bible_cer"], "pages": r["pages"]}
        if k == "ood":  # v2: pages that switch passage (lectionaries) are scored per passage
            with contextlib.redirect_stdout(io.StringIO()):
                res[k]["bible_cer_v2"] = bible_ref.score_segmented(str(d), str(refs))["bible_cer_v2"]
    assert i == len(flat)
    (run_dir / "eval_suite.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
