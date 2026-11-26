"""Inference helpers for CRNN checkpoints (train_crnn.py) with optional char-LM decoding."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import charlm  # noqa: E402
from train_crnn import CRNN, Codec  # noqa: E402

DEFAULT_LM = dict(alpha=0.35, beta=0.5, beam=12)  # tuned on SCAM val for E09-family models


class Recognizer:
    """ckpt may be a comma-separated list: the models' CTC probabilities are averaged
    (all must share codec and input height)."""

    def __init__(self, ckpt, device=None, lm_path=None, alpha=None, beta=None, beam=None):
        ckpts = [c for c in str(ckpt).split(",") if c]
        self.members = [_Single(c, device) for c in ckpts]
        first = self.members[0]
        for m in self.members[1:]:
            assert m.codec.chars == first.codec.chars and m.height == first.height, "incompatible ensemble members"
        self.device, self.codec, self.height = first.device, first.codec, first.height
        self.lm = charlm.load(lm_path) if lm_path else None
        self.alpha = DEFAULT_LM["alpha"] if alpha is None else alpha
        self.beta = DEFAULT_LM["beta"] if beta is None else beta
        self.beam = DEFAULT_LM["beam"] if beam is None else beam

    def probs(self, images, batch=32):
        outs = [m.probs(images, batch) for m in self.members]
        if len(outs) == 1:
            return outs[0]
        return [sum(ps) / len(ps) for ps in zip(*outs)]

    def decode(self, prob):
        if self.lm is None:
            return self.codec.decode_greedy(prob.argmax(0).tolist())
        return charlm.ctc_beam_search(prob, self.codec.labels, self.lm, self.alpha, self.beta, self.beam)

    def __call__(self, images):
        return [self.decode(p) for p in self.probs(images)]


class _Single:
    def __init__(self, ckpt, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if str(ckpt).endswith(".safetensors"):  # published format: weights + <name>.json (codec, args)
            from safetensors.torch import load_file
            ck = json.loads(Path(ckpt).with_suffix(".json").read_text(encoding="utf-8"))
            ck["model"] = load_file(ckpt)
        else:
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        a = ck["args"]
        self.codec = Codec(ck["codec"])
        self.height, self.max_width = a["height"], a["max_width"]
        self.model = CRNN(len(self.codec.chars) + 1, a["backbone"], pretrained=False, hidden=a["hidden"],
                          dropout=0.0)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()

    def _tensor(self, im: Image.Image):
        im = im.convert("L")
        w, h = im.size
        nw = max(16, min(self.max_width, int(round(w * self.height / h))))
        im = im.resize((nw, self.height), Image.BILINEAR)
        return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None]

    @torch.no_grad()
    def probs(self, images, batch=32):
        """Returns list of (C, T) numpy probability matrices."""
        out = []
        for i in range(0, len(images), batch):
            xs = [self._tensor(im) for im in images[i:i + batch]]
            W = max(x.shape[2] for x in xs)
            x = torch.ones(len(xs), 1, self.height, W)
            for j, t in enumerate(xs):
                x[j, :, :, : t.shape[2]] = t
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                logits = self.model(x.to(self.device))
            p = logits.float().softmax(-1).cpu()
            ol = CRNN.out_lengths(torch.tensor([t.shape[2] for t in xs]))
            out += [p[j, : ol[j]].T.numpy() for j in range(len(xs))]
        return out

