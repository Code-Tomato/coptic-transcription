"""Finalize a Kraken training run directory.

1. pick the checkpoint with the best validation accuracy (encoded in its filename)
2. export it to <run>/best.safetensors
3. score it on SCAM-A val with our metric -> <run>/scores_val.json, <run>/preds_val.tsv
4. optionally score test (--test) -> <run>/scores_test.json
5. delete the remaining checkpoints (--prune) to save disk

    python scripts/finish_kraken_run.py runs/E01_kraken_scam_only --prune
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv/bin/python"
KETOS = ROOT / ".venv/bin/ketos"


def best_ckpt(run: Path) -> Path:
    best, best_acc = None, -1.0
    for c in (run / "checkpoints").glob("checkpoint_*.ckpt"):
        m = re.search(r"checkpoint_(\d+)-([0-9.]+)\.ckpt$", c.name)
        if m and float(m.group(2)) > best_acc:
            best, best_acc = c, float(m.group(2))
    if best is None:
        raise SystemExit(f"no checkpoints in {run}")
    print(f"best checkpoint: {best.name} (kraken val acc {best_acc})")
    return best


def evaluate(model: Path, lst: str, out_prefix: Path):
    r = subprocess.run([str(PY), "-I", str(ROOT / "scripts/eval_kraken.py"), str(model), lst,
                        "--out", str(out_prefix) + ".tsv"], capture_output=True, text=True, check=True)
    res = json.loads(r.stdout.strip().splitlines()[-1])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--ckpt", help="use this checkpoint instead of the best one")
    args = ap.parse_args()
    run = Path(args.run)
    ck = Path(args.ckpt) if args.ckpt else best_ckpt(run)
    model = run / "best.safetensors"
    subprocess.run([str(KETOS), "convert", "-o", str(model), str(ck)], check=True, capture_output=True)
    (run / "best_checkpoint.txt").write_text(ck.name + "\n")
    val = evaluate(model, str(ROOT / "data/scam_a/val.lst"), run / "preds_val")
    (run / "scores_val.json").write_text(json.dumps(val, indent=1))
    print("val", val)
    if args.test:
        test = evaluate(model, str(ROOT / "data/scam_a/test.lst"), run / "preds_test")
        (run / "scores_test.json").write_text(json.dumps(test, indent=1))
        print("test", test)
    if args.prune:
        for c in (run / "checkpoints").glob("*.ckpt"):
            if c != ck:
                c.unlink()


if __name__ == "__main__":
    sys.exit(main())
