<img src="assets/icon.svg" width="64" height="64" alt="">

# Coptic transcription

Two tools for working with Coptic:

1. **Hymn sing-along** ([`liturgy/`](liturgy/README.md)). Pairs a recording of a hymn with its text, so each word lights up as it is sung. The text shows in Coptic, English letters, English and Arabic.
   - **Demo:** **https://code-tomato.github.io/coptic-transcription/**
   - **Accuracy:** 93% of words start within 0.5 s of hand-checked timings across tasbeha.org's 75 timed recordings.
2. **Manuscript and print OCR** (the rest of this README). Reads Sahidic Coptic from photos or scans of whole pages.
   - **Accuracy:** 11.5% character error rate on SCAM-A test lines, and 7.0% on whole test pages.
   - **Models:** on Hugging Face at [`codetomato/coptic-ocr`](https://huggingface.co/codetomato/coptic-ocr).

Code is MIT-licensed ([LICENSE](LICENSE)). Data, recordings, texts and manuscript images aren't included; they belong to their owners (see the credits in each README). Rebuild the demo site with `python liturgy/scripts/build_site.py`, which writes to `docs/`.

## OCR: how it was built

The approach is **hillclimbing**: one fixed benchmark, one number (CER), and a log of every experiment in
[EXPERIMENTS.md](EXPERIMENTS.md). An idea is kept only if it lowers validation CER. The test split is
scored only for a model we'd stand behind.

## Benchmark

| Track | Data | Split | Status |
|---|---|---|---|
| **A. Manuscripts (main)** | SCAM-A: 24 Vatican pages of a White Monastery Sahidic codex (Borg. copt. 109), 1,445 lines | SCAM's own: train 943 / val 242 / test 260 lines, split by page | ready |
| B. Printed editions | to be built (see "Resources") | — | planned |

- **Metric:** CER (character error rate, %) over grapheme clusters (a letter together with its
  combining marks counts as one unit). Like the SCAM paper, it's summed over the whole split.
  Code-point CER and SER (% of lines with any error) are reported too.
- **Normalization** (`scripts/textnorm.py`): NFC, NBSP and tabs become spaces, whitespace runs are
  collapsed, ends trimmed. Applied to ground truth and predictions alike.
- **Published reference points on SCAM-A** (trained on A, tested on A; arXiv 2606.15987, Table 3):
  VAN 7.50, CRNN 13.91, TrOCR-L 14.99, TrOCR-S 30.70.

## Resources found

| Resource | What it is | Use here | License |
|---|---|---|---|
| [SCAM](https://huggingface.co/datasets/blowing-up-groundhogs/scam) ([paper](https://arxiv.org/abs/2606.15987)) | Line polygons and transcriptions for 53 pages of a Sahidic manuscript | Real handwritten training and test data. SCAM-A images come from [Vatican IIIF](https://digi.vatlib.it/mss/Borg.copt); SCAM-B images must be requested from the Naples library | CC-BY 4.0 (annotations); images © BAV |
| [Coptic Scriptorium corpora](https://github.com/CopticScriptorium/corpora) | about 80 corpora of Sahidic and Bohairic text, many with manuscript line breaks (`lb_n`) and diplomatic spellings | Text source for synthetic lines; character language model | CC-BY 4.0 (mostly) |
| [KELLIA/CopticOCR](https://github.com/KELLIA/CopticOCR) | OCRopus models for printed Coptic (Besa, Apophthegmata, Bohairic Minor Prophets) | Printed-text reference. The models are Python 2 OCRopus | CC-BY-SA 4.0 |
| [tessdata_coptic](https://github.com/Shreeshrii/tessdata_coptic) | Tesseract 4 `cop.traineddata` | Printed-text baseline | Apache-2.0 |
| Coptic OCR ground truth (Lincke et al., DATeCH 2019) | Printed-edition GT at `vcs.etrap.eu/Coptic-OCR/datasets` | Would be ideal for track B, but **the host doesn't resolve** (checked 2026-11-21) | ? |
| [Vatican Borg. copt. 109](https://digi.vatlib.it/mss/Borg.copt) (unlabeled) | 160 digitized fascicles of White Monastery leaves, many scribes | Pseudo-labels (self-training) and an out-of-domain benchmark (biblical leaves aligned to the Sahidic Bible) | © BAV, research use |
| Sahidic NT and OT (Coptic Scriptorium `sahidica.nt`, `sahidic.ot`) | Bible text | References for the out-of-domain benchmark. **Not** used in the LM or synthetic data | CC-BY |
| [Horner, Sahidic NT (1911–24)](https://archive.org/details/copticversionofn02hornuoft) | Printed edition, public domain | Printed-text benchmark (main text above the apparatus rule) | PD |
| Fonts: Antinoou, New Athena Unicode, Noto Sans Coptic, FreeSerif | Unicode Coptic fonts | Rendering synthetic lines | OFL / free |

## Use it

```bash
# whole page (photo or scan): segment -> read lines -> Coptic text, one line per text line
.venv/bin/python scripts/transcribe.py page.jpg --overlay boxes.jpg
# images that are already single text lines
.venv/bin/python scripts/transcribe.py --lines line1.png line2.png
```
Models are read from `models/`. Download them from [Hugging Face](https://huggingface.co/codetomato/coptic-ocr), where they're stored as safetensors (`--rec models/recognizer.safetensors`):
- `recognizer.pt`: ResNet-CRNN
- `segmenter.safetensors`: Kraken BLLA fine-tuned on SCAM pages
- `coptic_char5.pkl`: character 5-gram LM

`--no-lm` turns off LM decoding.

## Where things stand

See [EXPERIMENTS.md](EXPERIMENTS.md) for every run. Shipped models:
- `models/recognizer.pt`: **E17s1** (default). ResNet50 + BiLSTM-512, trained on everything at once, with simulated bleed-through.
- `models/recognizer_manuscript.pt`: E10a. The earlier manuscript specialist; much worse on print.
- `models/recognizer_E12.pt`: the previous default, kept for comparison.

| Benchmark | Metric | E17s1 (default) | E10a | E12 (previous default) | Reference |
|---|---|---|---|---|---|
| SCAM-A **test** lines (same scribe as gold training data) | CER (official) | **11.52** | 11.99 | 12.52 | paper: VAN 7.50, CRNN 13.91 |
| same, spaces ignored (one test page uses word spaces, see log) | CER | **8.01** | 8.50 | 9.02 | |
| SCAM-A **test**, full pages (auto segmentation + reading order) | page CER | 7.04 | **6.69** | 7.52 | |
| Held-out biblical leaves, **35 pages by unseen scribes** | letters CER vs. edition (v2)* | 10.99 | **10.98** | 11.16 | E02 (SCAM only): 16.15 |
| Printed: Horner Sahidic NT, Luke (14 pages) | letters CER vs. edition* | **18.11** | 30.62 | 18.47 | before printed data: 27.66 |

\* Includes an unknown floor from spelling differences between each manuscript or print and the reference edition.
v2 scores lectionary pages per passage. v1 charged each jump between books as errors, which added about 9 points; see the log.

What moved the numbers, in order:
1. **ImageNet-pretrained ResNet backbone** instead of Kraken from scratch: 88.8 → 8.8 val CER.
2. **Fine-tuning the segmenter** on SCAM pages plus column-aware reading order: page CER 42 → 10.
3. **Character LM** from Coptic Scriptorium: −0.5 to −1.0 CER.
4. **Synthetic lines** (fonts on real parchment): −0.95.
5. **Pseudo-labels** from about 1,100 unlabeled Vatican pages: unseen-scribe CER 16.2 → 11.0 (v2).
6. **Alignment-based printed ground truth** from Horner: printed CER 27.7 → 18.5.
7. **Bigger backbone (ResNet50) and simulated bleed-through**, trained on rented GPUs: val CER with LM 7.85 → 6.98, test 11.99 → 11.52.

## Training on a rented GPU

Training runs on Vast.ai, not on this laptop. The laptop is node1's internet gateway, and a heat crash took both offline.
The `vastai` CLI and API key are set up on omarchy. A round is one shell script (`runs/cloud_C*.sh`):
1. Rent a cheap 24 GB card (an RTX 3090 at about $0.18/h was plenty; training is limited by data loading) with `--label coptic-train`.
2. Upload code and the needed data at the same absolute path (about 2.4 GB; lists hold absolute paths).
3. Run the script in `tmux`. Score with `scripts/eval_suite.py` (val, unseen scribes and print in one pass; `--test` only for finalists).
4. Copy `runs/<name>/` back, then `vastai destroy instance <id> -y`.

## Layout

```
scripts/        all code (data prep, synthetic data, training, evaluation)
data/           downloads and generated data (git-ignored; rebuilt by scripts)
runs/<name>/    one folder per experiment: config, logs, scores (checkpoints are git-ignored)
EXPERIMENTS.md  the hillclimb log
```
