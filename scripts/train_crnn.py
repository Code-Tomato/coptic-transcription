"""Line recognizer: ImageNet-pretrained ResNet + BiLSTM + CTC.

    python scripts/train_crnn.py --name E04_resnet_scam --train data/scam_a/train.lst \
        [--synth data/synth/v1 --synth-ratio 0.5] [--init runs/X/best.pt] --steps 6000

Writes runs/<name>/{config.json, log.jsonl, best.pt, preds_val.tsv, probs_val.pkl, scores_val.json}
Validation (SCAM-A val, greedy CTC) runs every --eval-every steps; best.pt keeps the lowest val CER.
"""
import argparse
import json
import math
import pickle
import random
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from metrics import score  # noqa: E402
from textnorm import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ data
def gt_of(img: str) -> str:
    return normalize(Path(img[: img.rfind(".")] + ".gt.txt").read_text(encoding="utf-8"))


def read_list(path):
    return [l.strip() for l in open(path) if l.strip()]


class Codec:
    """Code-point level codec; label 0 is the CTC blank."""

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.c2i = {c: i + 1 for i, c in enumerate(self.chars)}

    def encode(self, s):
        return [self.c2i[c] for c in s if c in self.c2i]

    def decode_greedy(self, ids):
        out, prev = [], 0
        for i in ids:
            if i != prev and i != 0:
                out.append(self.chars[i - 1])
            prev = i
        return "".join(out)

    @property
    def labels(self):
        return {i + 1: c for i, c in enumerate(self.chars)}


def augment_real(im: Image.Image, rng: random.Random, strength: float) -> Image.Image:
    """Geometric + photometric augmentation for real manuscript lines."""
    if strength <= 0:
        return im
    w, h = im.size
    # shear / rotation / scale through an affine transform
    sh = rng.uniform(-0.2, 0.2) * strength
    ang = math.radians(rng.uniform(-2.5, 2.5) * strength)
    sx = 1 + rng.uniform(-0.15, 0.15) * strength
    sy = 1 + rng.uniform(-0.12, 0.12) * strength
    a, b = sx * math.cos(ang), sx * (math.sin(ang) + sh)
    c, d = -sy * math.sin(ang), sy * math.cos(ang)
    det = a * d - b * c
    ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det
    cx, cy = w / 2, h / 2
    tx = cx - (ia * cx + ib * cy) + rng.uniform(-0.03, 0.03) * w * strength
    ty = cy - (ic * cx + id_ * cy) + rng.uniform(-0.06, 0.06) * h * strength
    fill = tuple(int(v) for v in np.median(np.asarray(im).reshape(-1, len(im.getbands())), axis=0))
    im = im.transform((w, h), Image.AFFINE, (ia, ib, tx, ic, id_, ty), resample=Image.BILINEAR,
                      fillcolor=fill if len(fill) > 1 else fill[0])
    if rng.random() < 0.3 * strength:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    if rng.random() < 0.2 * strength:
        k = rng.choice([ImageFilter.MinFilter(3), ImageFilter.MaxFilter(3)])
        im = im.filter(k)
    # random crop of a few pixels at each side (polygon crops vary)
    if rng.random() < 0.5:
        l = rng.randint(0, max(1, int(0.03 * w))); r = rng.randint(0, max(1, int(0.03 * w)))
        t = rng.randint(0, max(1, int(0.08 * h))); bt = rng.randint(0, max(1, int(0.08 * h)))
        im = im.crop((l, t, w - r, h - bt))
    return im


class LineDataset(Dataset):
    def __init__(self, items, height, max_width, augment=0.0, photometric=True, seed=0, showthrough=0.0):
        self.items = items  # list of (img_path, text)
        self.height, self.max_width = height, max_width
        self.augment, self.photometric = augment, photometric
        self.seed = seed
        self.showthrough = showthrough  # probability of blending in mirrored text from another line

    def _ghost(self, rng, nw):
        """Ink from the other side of the leaf: another line, mirrored, blurred, faded; values in [0,1], 1 = paper."""
        p, _ = self.items[rng.randrange(len(self.items))]
        g = Image.open(p).convert("L").transpose(Image.FLIP_LEFT_RIGHT)
        g = g.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.0)))
        gw = max(16, int(round(g.width * self.height / g.height)))
        g = g.resize((gw, self.height), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(g, dtype=np.float32) / 255.0)
        t = (t - t.median()).clamp(max=0) + 1  # keep only ink darker than the paper
        dy = rng.randint(-self.height // 3, self.height // 3)
        t = torch.roll(t, dy, 0)
        if gw < nw:
            t = torch.cat([t] * (nw // gw + 1), 1)
        x0 = rng.randint(0, t.shape[1] - nw)
        return t[:, x0:x0 + nw]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, text = self.items[i]
        rng = random.Random()
        im = Image.open(path).convert("L")
        im = augment_real(im, rng, self.augment)
        w, h = im.size
        nw = max(16, min(self.max_width, int(round(w * self.height / h))))
        im = im.resize((nw, self.height), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None]
        if self.showthrough > 0 and rng.random() < self.showthrough:
            x = x * (1 - rng.uniform(0.25, 0.6) * (1 - self._ghost(rng, nw)))[None]
        if self.augment > 0 and self.photometric:
            # contrast / brightness / gamma / noise
            g = math.exp(rng.uniform(-0.4, 0.4) * self.augment)
            x = x.clamp(1e-4, 1) ** g
            x = (x - x.mean()) * rng.uniform(0.7, 1.3) + x.mean() + rng.uniform(-0.1, 0.1) * self.augment
            if rng.random() < 0.3:
                x = x + torch.randn_like(x) * rng.uniform(0.01, 0.05)
            if rng.random() < 0.2 * self.augment:  # erase a small vertical strip
                sw = rng.randint(2, max(3, nw // 25)); sx = rng.randint(0, max(0, nw - sw))
                x[..., sx:sx + sw] = x.mean()
        x = x.clamp(0, 1)
        return x, text, path


def collate(batch):
    xs, texts, paths = zip(*batch)
    H = xs[0].shape[1]
    W = max(x.shape[2] for x in xs)
    out = torch.ones(len(xs), 1, H, W)
    widths = []
    for i, x in enumerate(xs):
        out[i, :, :, : x.shape[2]] = x
        widths.append(x.shape[2])
    return out, list(texts), list(paths), torch.tensor(widths)


class MixedSampler(torch.utils.data.Sampler):
    """Infinite sampler drawing from real items with prob (1-ratio) and synth with prob ratio."""

    def __init__(self, n_real, n_synth, ratio, seed=0):
        self.n_real, self.n_synth, self.ratio = n_real, n_synth, ratio
        self.rng = random.Random(seed)

    def __iter__(self):
        while True:
            if self.n_synth and self.rng.random() < self.ratio:
                yield self.n_real + self.rng.randrange(self.n_synth)
            else:
                yield self.rng.randrange(self.n_real)

    def __len__(self):
        return 1 << 40


class ConcatItems(Dataset):
    def __init__(self, real, synth):
        self.real, self.synth = real, synth

    def __len__(self):
        return len(self.real) + (len(self.synth) if self.synth else 0)

    def __getitem__(self, i):
        return self.real[i] if i < len(self.real) else self.synth[i - len(self.real)]


# ------------------------------------------------------------------ model
class CRNN(nn.Module):
    def __init__(self, n_classes, backbone="resnet34", pretrained=True, hidden=256, layers=2, dropout=0.2):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models, backbone)(weights=weights)
        # keep conv1..layer3; make layer3 stride (2,1) so width keeps 1/8 resolution
        b0 = net.layer3[0]
        bottleneck = hasattr(b0, "conv3")  # resnet50+: torchvision puts the stride on conv2
        (b0.conv2 if bottleneck else b0.conv1).stride = (2, 1)
        b0.downsample[0].stride = (2, 1)
        self.cnn = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3)
        last = net.layer3[-1]
        ch = (last.bn3 if bottleneck else last.bn2).num_features
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(ch, hidden, num_layers=layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(2 * hidden, n_classes)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):  # x: (N,1,H,W) in [0,1]
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        f = self.cnn(x)               # (N, C, H', W/8)
        f = f.mean(2).transpose(1, 2)  # (N, T, C)
        f, _ = self.rnn(self.drop(f))
        return self.fc(self.drop(f))   # (N, T, K)

    @staticmethod
    def out_lengths(widths):
        return torch.div(widths + 7, 8, rounding_mode="floor").clamp(min=1)


# ------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate(model, loader, codec, device, keep_probs=False):
    model.eval()
    pairs, rows, probs_out = [], [], []
    for x, texts, paths, widths in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x.to(device))
        prob = logits.float().softmax(-1).cpu()
        olens = CRNN.out_lengths(widths)
        for i in range(len(texts)):
            p = prob[i, : olens[i]]
            pred = codec.decode_greedy(p.argmax(-1).tolist())
            pairs.append((pred, texts[i]))
            rows.append(f"{Path(paths[i]).name}\t{texts[i]}\t{pred}")
            if keep_probs:
                probs_out.append((paths[i], p.T.numpy()))  # (C, T) like kraken
    model.train()
    return score(pairs), rows, probs_out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", default=str(ROOT / "data/scam_a/train.lst"))
    ap.add_argument("--val", default=str(ROOT / "data/scam_a/val.lst"))
    ap.add_argument("--synth", nargs="*", default=[], help="dirs of synthetic <id>.png/<id>.gt.txt")
    ap.add_argument("--synth-ratio", type=float, default=0.0)
    ap.add_argument("--init", help="checkpoint to start from (codec is merged)")
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--max-width", type=int, default=1200)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--augment", type=float, default=1.0)
    ap.add_argument("--showthrough", type=float, default=0.0, help="prob. of simulated bleed-through on real lines")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    run = ROOT / "runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real = [(p, gt_of(p)) for p in read_list(args.train)]
    synth = []
    for d in args.synth:
        for p in sorted(Path(d).glob("*.png")):
            synth.append((str(p), gt_of(str(p))))
    val = [(p, gt_of(p)) for p in read_list(args.val)]

    chars = set("".join(t for _, t in real + synth))
    init = None
    if args.init:
        init = torch.load(args.init, map_location="cpu", weights_only=False)
        chars |= set(init["codec"])
    codec = Codec(chars)
    print(f"real {len(real)}  synth {len(synth)}  val {len(val)}  classes {len(codec.chars)+1}", flush=True)

    model = CRNN(len(codec.chars) + 1, args.backbone, not args.no_pretrained, args.hidden, dropout=args.dropout)
    if init is not None:
        sd = init["model"]
        old = init["codec"]
        # remap output layer rows to the new codec
        W, b = sd.pop("fc.weight"), sd.pop("fc.bias")
        model.load_state_dict(sd, strict=False)
        with torch.no_grad():
            model.fc.weight[0], model.fc.bias[0] = W[0], b[0]
            for j, c in enumerate(old):
                if c in codec.c2i:
                    model.fc.weight[codec.c2i[c]] = W[j + 1]
                    model.fc.bias[codec.c2i[c]] = b[j + 1]
    model.to(device)

    ds_real = LineDataset(real, args.height, args.max_width, augment=args.augment, showthrough=args.showthrough)
    ds_synth = LineDataset(synth, args.height, args.max_width, augment=args.augment * 0.5) if synth else None
    ds = ConcatItems(ds_real, ds_synth)
    sampler = MixedSampler(len(real), len(synth), args.synth_ratio if synth else 0.0, args.seed)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, collate_fn=collate, num_workers=args.workers,
                    persistent_workers=True, prefetch_factor=4)
    vdl = DataLoader(LineDataset(val, args.height, args.max_width), batch_size=32, collate_fn=collate, num_workers=2)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    best = (1e9, -1)
    log = open(run / "log.jsonl", "a")
    t0 = time.time()
    it = iter(dl)
    running = 0.0
    for step in range(1, args.steps + 1):
        x, texts, _, widths = next(it)
        targets = [torch.tensor(codec.encode(t), dtype=torch.long) for t in texts]
        tl = torch.tensor([len(t) for t in targets])
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
        lp = logits.float().log_softmax(-1).transpose(0, 1)  # (T, N, K)
        ol = CRNN.out_lengths(widths).clamp(max=lp.shape[0])
        loss = ctc(lp, torch.cat(targets), ol, tl)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update(); sched.step()
        running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()

        if step % args.eval_every == 0 or step == args.steps:
            s, rows, _ = evaluate(model, vdl, codec, device)
            rec = {"step": step, "loss": round(running, 4), "lr": sched.get_last_lr()[0],
                   "val_cer": round(s["cer"], 2), "val_ser": round(s["ser"], 2), "min": round((time.time() - t0) / 60, 1)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if s["cer"] < best[0]:
                best = (s["cer"], step)
                torch.save({"model": model.state_dict(), "codec": codec.chars, "args": vars(args), "step": step,
                            "val": s}, run / "best.pt")

    # final: reload best, write predictions + probabilities for LM sweeps
    ck = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    s, rows, probs = evaluate(model, vdl, codec, device, keep_probs=True)
    (run / "preds_val.tsv").write_text("image\tgt\tpred\n" + "\n".join(rows) + "\n", encoding="utf-8")
    with open(run / "probs_val.pkl", "wb") as f:
        pickle.dump({"labels": codec.labels, "lines": probs}, f)
    s["best_step"] = best[1]
    (run / "scores_val.json").write_text(json.dumps(s, indent=1))
    print("BEST", json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
