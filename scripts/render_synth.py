"""Render synthetic Coptic text-line images for pretraining.

Text: Coptic Scriptorium manuscript lines converted to SCAM conventions.
Backgrounds: real parchment patches cut from SCAM-A *train* pages (areas
outside every annotated line), mixed with procedural parchment.
Output: Kraken "path" format (<id>.png + <id>.gt.txt) in --out.

    python scripts/render_synth.py --lines data/text/cs_lines.tsv \
        --fonts data/fonts/use --scam-xml ... --scam-images ... --out data/synth/v1 -n 50000
"""
import argparse
import math
import random
import sys
import unicodedata
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

sys.path.insert(0, str(Path(__file__).parent))
from scam_convention import is_allowed, to_scam  # noqa: E402

LAYOUT = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC


# ---------------------------------------------------------------- text
def load_texts(path):
    texts = []
    for row in open(path, encoding="utf-8"):
        t = to_scam(row.rstrip("\n").split("\t")[-1])
        if is_allowed(t) and 2 <= len(t) <= 40:
            texts.append(t)
    return texts


def sample_text(texts, rng):
    r = rng.random()
    if r < 0.6:
        return rng.choice(texts)
    # splice neighbouring material to get new line breaks / lengths
    s = "".join(rng.choice(texts) for _ in range(3))
    base = [i for i, ch in enumerate(s) if not unicodedata.combining(ch)]
    n = rng.randint(4, 18)
    if len(base) <= n:
        return s
    start = rng.randrange(0, len(base) - n)
    a, b = base[start], base[start + n]
    return s[a:b].strip(" ")


# ---------------------------------------------------------------- fonts
class Font:
    def __init__(self, path):
        self.path = str(path)
        self.cmap = set(TTFont(path, lazy=True).getBestCmap().keys())

    def covers(self, text):
        return all(ord(ch) in self.cmap or ch == " " for ch in text)


# ---------------------------------------------------------------- backgrounds
def scam_background_patches(xml, images, rng, per_page=60, size=(360, 110)):
    patches = []
    root = ET.parse(xml).getroot()
    for im in root.iter("image"):
        if im.get("subset") != "train":
            continue
        page = Image.open(Path(images) / f"{im.get('name')}.jpg").convert("RGB")
        sx = page.width / float(im.get("width"))
        sy = page.height / float(im.get("height"))
        boxes = []
        for p in im.iter("polygon"):
            pts = [tuple(map(float, xy.split(","))) for xy in p.get("points").split(";")]
            xs = [x * sx for x, _ in pts]
            ys = [y * sy for _, y in pts]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        if not boxes:
            continue
        tx0 = min(b[0] for b in boxes); ty0 = min(b[1] for b in boxes)
        tx1 = max(b[2] for b in boxes); ty1 = max(b[3] for b in boxes)
        mx, my = 0.08 * (tx1 - tx0), 0.05 * (ty1 - ty0)
        got = 0
        for _ in range(per_page * 30):
            w, h = size[0] * rng.uniform(0.4, 1.0), size[1] * rng.uniform(0.4, 1.0)
            x = rng.uniform(max(0, tx0 - mx), min(page.width, tx1 + mx) - w)
            y = rng.uniform(max(0, ty0 - my), min(page.height, ty1 + my) - h)
            if any(x < b[2] + 4 and x + w > b[0] - 4 and y < b[3] + 4 and y + h > b[1] - 4 for b in boxes):
                continue
            patch = page.crop((int(x), int(y), int(x + w), int(y + h)))
            a = np.asarray(patch.convert("L"), dtype=np.float32)
            if a.mean() < 90 or a.std() > 28:  # skip dark scanner background / ink
                continue
            patches.append(patch)
            got += 1
            if got >= per_page:
                break
    return patches


def procedural_background(w, h, rng):
    base = np.array([rng.uniform(170, 235), rng.uniform(140, 215), rng.uniform(100, 185)])
    small = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 1, (max(2, h // 16), max(2, w // 16), 1))
    low = np.asarray(Image.fromarray(((small - small.min()) / (np.ptp(small) + 1e-6) * 255).astype(np.uint8)[..., 0]).resize((w, h), Image.BICUBIC), dtype=np.float32)[..., None] / 255.0
    fine = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 9), (h, w, 1))
    img = base[None, None, :] * (0.85 + 0.3 * low) + fine
    return np.clip(img, 0, 255)


# ---------------------------------------------------------------- rendering
def render_ink(text, font, size, rng):
    f = ImageFont.truetype(font.path, size, layout_engine=LAYOUT)
    l, t, r, b = f.getbbox(text)
    pad = int(size * 0.6)
    W, H = r - l + 2 * pad, int(size * 1.6) + 2 * pad
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    spacing = rng.uniform(-0.04, 0.12) * size
    if abs(spacing) > 1 and rng.random() < 0.5:
        # draw cluster by cluster to vary letter spacing
        x = pad
        clusters, cur = [], ""
        for ch in text:
            if cur and not unicodedata.combining(ch):
                clusters.append(cur); cur = ""
            cur += ch
        clusters.append(cur)
        for c in clusters:
            d.text((x, pad - t + size * 0.3), c, font=f, fill=255)
            x += f.getlength(c) + spacing
        W2 = int(x + pad)
        if W2 > W:
            m2 = Image.new("L", (W2, H), 0); m2.paste(mask, (0, 0)); mask = m2
            d = ImageDraw.Draw(mask)
            x = pad
            for c in clusters:
                d.text((x, pad - t + size * 0.3), c, font=f, fill=255)
                x += f.getlength(c) + spacing
    else:
        d.text((pad - l, pad - t + size * 0.3), text, font=f, fill=255)
    return mask


def augment_mask(mask, rng):
    # stroke weight
    k = rng.choice([0, 0, 0, 3, 3, -3])
    if k > 0:
        mask = mask.filter(ImageFilter.MaxFilter(k))
    elif k < 0:
        mask = mask.filter(ImageFilter.MinFilter(-k))
    # shear (slant) + small rotation
    sh = rng.uniform(-0.25, 0.25)
    W, H = mask.size
    mask = mask.transform((W + int(abs(sh) * H), H), Image.AFFINE,
                          (1, sh, -sh * H if sh > 0 else 0, 0, 1, 0), resample=Image.BILINEAR)
    mask = mask.rotate(rng.uniform(-2.0, 2.0), resample=Image.BILINEAR, expand=True)
    # vertical squash / stretch
    sy = rng.uniform(0.85, 1.15)
    mask = mask.resize((mask.width, max(8, int(mask.height * sy))), Image.BILINEAR)
    # ragged ink: multiply by low-frequency noise
    if rng.random() < 0.6:
        a = np.asarray(mask, dtype=np.float32)
        n = np.random.default_rng(rng.randrange(1 << 30)).uniform(0.65, 1.0, (max(2, a.shape[0] // 6), max(2, a.shape[1] // 6)))
        n = np.asarray(Image.fromarray((n * 255).astype(np.uint8)).resize(a.shape[::-1], Image.BILINEAR), dtype=np.float32) / 255
        mask = Image.fromarray(np.clip(a * n, 0, 255).astype(np.uint8))
    return mask


def make_sample(text, font, texts, fonts, patches, rng):
    size = rng.randint(36, 64)
    mask = augment_mask(render_ink(text, font, size, rng), rng)
    W, H = mask.size
    # neighbouring lines peeking in at top/bottom like real crops
    extra = Image.new("L", (W, H), 0)
    for dy in (-1, 1):
        if rng.random() < 0.5:
            other = rng.choice(texts)
            f2 = font if font.covers(other) else rng.choice(fonts)
            if f2.covers(other):
                m2 = render_ink(other, f2, size, rng)
                off = int(dy * size * rng.uniform(1.15, 1.6))
                extra.paste(m2.crop((0, 0, min(W, m2.width), m2.height)), (rng.randint(-20, 20), off), m2.crop((0, 0, min(W, m2.width), m2.height)))
    ink = np.maximum(np.asarray(mask, dtype=np.float32), np.asarray(extra, dtype=np.float32)) / 255.0

    # background
    if patches and rng.random() < 0.7:
        bg = rng.choice(patches)
        if rng.random() < 0.5:
            bg = bg.transpose(Image.FLIP_LEFT_RIGHT)
        bg = np.asarray(bg.resize((W, H), Image.BICUBIC), dtype=np.float32)
        bg = bg * rng.uniform(0.85, 1.12) + rng.uniform(-15, 15)
    else:
        bg = procedural_background(W, H, rng)
    # bleed-through: faint mirrored text from the other side
    if rng.random() < 0.3:
        other = rng.choice(texts)
        if font.covers(other):
            m3 = render_ink(other, font, size, rng).transpose(Image.FLIP_LEFT_RIGHT).resize((W, H))
            g = np.asarray(m3.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)[..., None] / 255
            bg = bg * (1 - rng.uniform(0.1, 0.3) * g)
    ink_col = np.array([rng.uniform(15, 90), rng.uniform(10, 60), rng.uniform(5, 45)])
    alpha = ink[..., None] * rng.uniform(0.65, 1.0)
    img = bg * (1 - alpha) + ink_col[None, None, :] * alpha
    # uneven illumination
    if rng.random() < 0.4:
        gx = np.linspace(rng.uniform(0.75, 1.1), rng.uniform(0.75, 1.1), W)[None, :, None]
        img = img * gx
    img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    if rng.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.6)))
    # crop tight-ish vertically with random padding, like polygon bbox crops
    a = np.asarray(mask) > 60
    rows = np.where(a.any(1))[0]; cols = np.where(a.any(0))[0]
    if len(rows) and len(cols):
        h = rows[-1] - rows[0]
        y0 = max(0, rows[0] - int(h * rng.uniform(0.05, 0.3)))
        y1 = min(H, rows[-1] + int(h * rng.uniform(0.05, 0.3)))
        x0 = max(0, cols[0] - rng.randint(2, 25))
        x1 = min(W, cols[-1] + rng.randint(2, 25))
        img = img.crop((x0, y0, x1, y1))
    # downscale (resolution variety), sensor noise, jpeg
    scale = rng.uniform(0.45, 1.0)
    img = img.resize((max(8, int(img.width * scale)), max(8, int(img.height * scale))), Image.BILINEAR)
    if rng.random() < 0.5:
        arr = np.asarray(img, dtype=np.float32) + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(1, 8), (img.height, img.width, 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


_G = {}


def _init(lines, font_dir, xml, images, seed):
    rng = random.Random(seed)
    _G["texts"] = load_texts(lines)
    _G["fonts"] = [Font(p) for p in sorted(Path(font_dir).iterdir()) if p.suffix.lower() in (".ttf", ".otf")]
    _G["patches"] = scam_background_patches(xml, images, rng) if xml else []


def _work(args):
    idx, seed, out = args
    rng = random.Random(seed)
    texts, fonts, patches = _G["texts"], _G["fonts"], _G["patches"]
    for _ in range(20):
        text = sample_text(texts, rng)
        if not text or not is_allowed(text):
            continue
        cands = [f for f in fonts if f.covers(text)]
        if cands:
            break
    else:
        return None
    font = rng.choice(cands)
    img = make_sample(text, font, texts, fonts, patches, rng)
    if rng.random() < 0.5:
        img.convert("L").save(Path(out) / f"{idx:07d}.png")
    else:
        img.save(Path(out) / f"{idx:07d}.png")
    (Path(out) / f"{idx:07d}.gt.txt").write_text(text, encoding="utf-8")
    return Path(font.path).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--scam-xml")
    ap.add_argument("--scam-images")
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print("layout engine:", "raqm" if LAYOUT == ImageFont.Layout.RAQM else "basic")
    jobs = [(i, args.seed * 10_000_000 + i, args.out) for i in range(args.n)]
    from collections import Counter
    c = Counter()
    with Pool(args.workers, _init, (args.lines, args.fonts, args.scam_xml, args.scam_images, args.seed)) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, jobs, chunksize=64)):
            c[r] += 1
            if i % 5000 == 0:
                print(i, flush=True)
    print(c)


if __name__ == "__main__":
    main()
