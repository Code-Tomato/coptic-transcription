"""Score a Kraken recognition model on a line list with our own metrics.

    python scripts/eval_kraken.py MODEL data/scam_a/val.lst [--out preds.tsv]

Each list entry is a line image with a sibling <stem>.gt.txt.
Prints a JSON dict: cer (grapheme-cluster CER %), cer_codepoint, ser, lines.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from metrics import score  # noqa: E402


def gt_path(img: str) -> Path:
    return Path(img[: img.rfind(".")] + ".gt.txt")


def predict_lines(model_path, images, device="cuda:0"):
    from kraken.configs import RecognitionInferenceConfig
    from kraken.containers import BBoxLine, Segmentation
    from kraken.tasks import RecognitionTaskModel

    model = RecognitionTaskModel.load_model(model_path)
    cfg = RecognitionInferenceConfig(device=device) if "device" in RecognitionInferenceConfig.__init__.__code__.co_varnames else RecognitionInferenceConfig()
    try:
        model = model.to(device)
    except Exception:
        pass
    for img in images:
        im = Image.open(img)
        seg = Segmentation(type="bbox", imagename=img, text_direction="horizontal-lr",
                           script_detection=False,
                           lines=[BBoxLine(id="l0", bbox=(0, 0, im.width, im.height))])
        recs = list(model.predict(im, seg, cfg))
        probs = model.net.outputs[0].float().cpu().numpy()  # (C, T) softmax outputs
        yield img, (recs[0].prediction if recs else ""), probs, model.net.codec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("lst")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dump-probs", help="save per-line CTC probabilities (pickle) for LM sweeps")
    ap.add_argument("--lm", help="char LM pickle; decode with CTC beam search + LM")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--beam", type=int, default=16)
    args = ap.parse_args()
    images = [l.strip() for l in open(args.lst) if l.strip()]
    lm = None
    if args.lm:
        import charlm
        lm = charlm.load(args.lm)
    pairs, rows, dump = [], [], []
    for img, pred, probs, codec in predict_lines(args.model, images, args.device):
        labels = {k[0]: v for k, v in codec.l2c.items() if len(k) == 1}
        if args.dump_probs:
            dump.append((img, probs))
        if lm is not None:
            import charlm
            pred = charlm.ctc_beam_search(probs, labels, lm, args.alpha, args.beta, args.beam)
        gt = gt_path(img).read_text(encoding="utf-8")
        pairs.append((pred, gt))
        rows.append(f"{Path(img).name}\t{gt}\t{pred}")
    if args.dump_probs:
        import pickle
        with open(args.dump_probs, "wb") as f:
            pickle.dump({"labels": labels, "lines": dump}, f)
    if args.out:
        Path(args.out).write_text("image\tgt\tpred\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in score(pairs).items()}))


if __name__ == "__main__":
    main()
