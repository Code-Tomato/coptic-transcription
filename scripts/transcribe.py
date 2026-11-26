"""Transcribe Coptic from a photo or scan of a page.

    .venv/bin/python scripts/transcribe.py page.jpg [more.jpg ...] \
        [--rec models/recognizer.pt] [--seg models/segmenter.safetensors] [--no-lm] [--overlay out.jpg]

Pipeline: baseline segmentation (Kraken BLLA, fine-tuned on SCAM pages) ->
line crops -> ResNet-CRNN recognizer -> CTC beam search with a Coptic character
LM. Prints one recognised line per text line, in reading order.
Use --lines if the input images are already single text lines.
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = {
    "rec": ROOT / "models/recognizer.pt",
    "seg": ROOT / "models/segmenter.safetensors",
    "lm": ROOT / "models/coptic_char5.pkl",
}


def load_segmenter(path, device):
    import segment_pages  # noqa: F401  (installs the fast polygonizer patch below)
    import kraken.lib.vgsl.spred as spred
    from kraken.configs import SegmentationInferenceConfig
    from kraken.tasks import SegmentationTaskModel

    spred.calculate_polygonal_environment = segment_pages.fast_polygons
    model = SegmentationTaskModel.load_model(str(path)) if path and Path(path).exists() else SegmentationTaskModel.load_model()
    if device.startswith("cuda"):
        cfg = SegmentationInferenceConfig(accelerator="gpu", device=[0])
    else:
        cfg = SegmentationInferenceConfig(accelerator="cpu")
    return model, cfg


def segment(model, cfg, im, max_side=2000):
    import torch
    scale = 1.0
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.BILINEAR)
    with torch.inference_mode():
        seg = model.predict(im, cfg)
    boxes = []
    for ln in seg.lines:
        xs = [x for x, _ in ln.boundary]
        ys = [y for _, y in ln.boundary]
        h = max(ys) - min(ys)
        if h < 8 or (max(xs) - min(xs)) < 2 * h:
            continue
        x0, x1 = max(0, min(xs) - 0.05 * h), min(im.width, max(xs) + 0.05 * h)
        y0, y1 = max(0, min(ys)), min(im.height, max(ys))
        boxes.append(tuple(int(v / scale) for v in (x0, y0, x1, y1)))
    from segment_pages import column_order
    return [boxes[i] for i in column_order(boxes, im.width / scale)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--rec", default=str(DEFAULTS["rec"]))
    ap.add_argument("--seg", default=str(DEFAULTS["seg"]))
    ap.add_argument("--lm", default=str(DEFAULTS["lm"]))
    ap.add_argument("--no-lm", action="store_true")
    ap.add_argument("--lines", action="store_true", help="inputs are already single-line images")
    ap.add_argument("--overlay", help="write an overlay image with line boxes (first page only)")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    from recognize import Recognizer
    rec = Recognizer(args.rec, device=args.device, lm_path=None if args.no_lm else args.lm)
    if args.lines:
        ims = [ImageOps.exif_transpose(Image.open(p)) for p in args.images]
        for p, t in zip(args.images, rec(ims)):
            print(f"{p}\t{t}")
        return
    seg_model, cfg = load_segmenter(args.seg, args.device)
    for n, path in enumerate(args.images):
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        boxes = segment(seg_model, cfg, im)
        texts = rec([im.crop(b) for b in boxes]) if boxes else []
        if len(args.images) > 1:
            print(f"=== {path}")
        for t in texts:
            print(t)
        if args.overlay and n == 0:
            ov = im.copy()
            d = ImageDraw.Draw(ov)
            for i, b in enumerate(boxes):
                d.rectangle(b, outline=(220, 30, 30), width=3)
                d.text((b[0], b[1] - 12), str(i + 1), fill=(220, 30, 30))
            ov.save(args.overlay)


if __name__ == "__main__":
    main()
