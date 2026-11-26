"""Fine-tune the MMS forced-aligner acoustic model on hand-timed chant segments (see ft_data.py).

    python liturgy/scripts/ft_train.py --name F01 --lr 2e-5 --steps 1500

CTC loss on segments cut at tasbeha's word starts, so the model learns which letters a stretch of chant
belongs to, including 30-60 s melismas. The CNN feature encoder stays frozen. Every --eval-every steps
the full alignment pipeline (make_lesson defaults) runs on the val recordings; the checkpoint with the
most val words within 0.5 s of tasbeha's start is kept in liturgy/models/<name>/.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from align import MODEL, SR, Aligner, load_audio, run_lesson, summary  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LES = ROOT / "data/tasbeha/lessons"
PIPE = dict(garbage=-1.02, line_breaks=True, full_text=True, skip_line=-9.0, blank_penalty=1.0, hold="vowels")


def batches(segs, max_sec, rng):
    segs = segs[:]
    rng.shuffle(segs)
    segs.sort(key=lambda s: s["end"] - s["start"] + rng.random() * 3)  # bucket by length, a little shuffled
    out, cur, longest = [], [], 0.0
    for s in segs:
        d = s["end"] - s["start"]
        if cur and max(longest, d) * (len(cur) + 1) > max_sec:
            out.append(cur)
            cur, longest = [], 0.0
        cur.append(s)
        longest = max(longest, d)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def validate(al, ids):
    errs, hold = [], []
    for i in ids:
        r = run_lesson(al, LES / f"{i}.json", LES / f"{i}.mp3", em=al.emissions(AUDIO[i]), **PIPE)
        errs += r["word_err"]
        hold += r["after_hold_err"]
    return summary(errs), summary(hold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--batch-sec", type=float, default=120.0, help="padded audio seconds per batch")
    ap.add_argument("--freeze-layers", type=int, default=0, help="also freeze the lowest N transformer layers")
    ap.add_argument("--mask-time-prob", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    out = ROOT / "models" / a.name
    out.mkdir(parents=True, exist_ok=True)

    segs = [json.loads(l) for l in open(ROOT / "data/ft/segments.jsonl")]
    train = [s for s in segs if s["split"] == "train"]
    val_ids = (ROOT / "data/ft/val_ids.txt").read_text().split()
    global AUDIO
    AUDIO = {i: load_audio(LES / f"{i}.mp3") for i in sorted({s["id"] for s in train} | set(val_ids))}
    print(f"{len(train)} train segments, {sum(s['end'] - s['start'] for s in train) / 60:.0f} min; val {val_ids}", flush=True)

    from transformers import Wav2Vec2ForCTC
    al = Aligner(device="cuda")
    model = Wav2Vec2ForCTC.from_pretrained(MODEL, ctc_loss_reduction="mean", ctc_zero_infinity=True,
                                           mask_time_prob=a.mask_time_prob, mask_time_length=10,
                                           layerdrop=0.0).cuda()
    with torch.no_grad():  # loaded uninitialised (not in the checkpoint); garbage here gives NaN losses
        model.wav2vec2.masked_spec_embed.uniform_()
    model.freeze_feature_encoder()
    for layer in model.wav2vec2.encoder.layers[: a.freeze_layers]:
        for p in layer.parameters():
            p.requires_grad = False
    model.gradient_checkpointing_enable()
    al._model = model
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * max(0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))))

    def evaluate(step):
        model.eval()
        w, h = validate(al, val_ids)
        model.train()
        return {"step": step, "val_words": w, "val_after_hold": h}

    log = [evaluate(0)]
    best = log[0]["val_words"]["within_0_5"]
    print(json.dumps(log[0]), flush=True)
    model.save_pretrained(out)  # step 0 = base model, in case nothing beats it
    step, t0, bad = 0, time.time(), 0
    model.train()
    while step < a.steps:
        for b in batches(train, a.batch_sec, rng):
            if step >= a.steps:
                break
            xs, ys = [], []
            for s in b:
                w = AUDIO[s["id"]][int(s["start"] * SR): int(s["end"] * SR)]
                w = (w - w.mean()) / (w.std() + 1e-7)
                xs.append(torch.from_numpy(w))
                ys.append(torch.tensor([al.vocab[c] for c in s["text"] if c in al.vocab]))
            L = max(len(x) for x in xs)
            x = torch.zeros(len(xs), L)
            mask = torch.zeros(len(xs), L, dtype=torch.long)
            for j, w in enumerate(xs):
                x[j, : len(w)] = w
                mask[j, : len(w)] = 1
            Y = max(len(y) for y in ys)
            y = torch.full((len(ys), Y), -100)
            for j, t in enumerate(ys):
                y[j, : len(t)] = t
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x.cuda(), attention_mask=mask.cuda(), labels=y.cuda()).loss
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                bad += 1
                print(f"step {step}: non-finite loss, batch skipped", flush=True)
                if bad > 20:
                    raise SystemExit("too many non-finite losses")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 20 == 0:
                print(f"step {step} loss {loss.item():.3f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                r = evaluate(step)
                log.append(r)
                cur = r["val_words"]["within_0_5"]
                print(json.dumps(r), flush=True)
                if cur > best:
                    best = cur
                    model.save_pretrained(out)
                    print(f"  saved (val {cur})", flush=True)
                (out / "train_log.json").write_text(json.dumps({"args": vars(a), "best_val_within_0_5": best, "log": log}, indent=1))
    print("best val within 0.5 s:", best)


AUDIO = {}

if __name__ == "__main__":
    main()
