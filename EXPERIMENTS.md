# Experiment log

Benchmark: SCAM-A line recognition. Lower CER is better. Val = 242 lines (4 pages),
test = 260 lines (4 pages). The test set is scored only for models selected on val.

| # | Date | Run | What changed | Val CER | Val SER | Test CER | Notes |
|---|---|---|---|---|---|---|---|
| — | — | paper | VAN (best published, A→A) | — | — | 7.50 | reference |
| — | — | paper | CRNN (A→A) | — | — | 13.91 | reference |
| E01 | 2026-11-21 | `runs/E01_kraken_scam_only` | Kraken 7.2 default VGSL (CNN+3×BiLSTM+CTC, from scratch), SCAM-A train only, `--augment` | 88.84 | 100.0 | — | Never got past the CTC "all blank" plateau. Train loss went 648→622 over 27 epochs; stopped at epoch 28. Training from scratch on 943 lines is too little. |
| E02 | 2026-11-21 | `runs/E02_resnet34_scam_only` | **Own trainer** (`train_crnn.py`): ImageNet ResNet34 (to layer3) + 2×BiLSTM-256 + CTC, grayscale, height 64, strong augmentation, AdamW 5e-4 cosine, 6k steps, SCAM-A train only | 8.76 | 54.1 | — | Pretrained backbone fixes convergence: 10.5 by step 1k. Overfits (train loss 0.02); best at step 5k. |
| E02+LM | 2026-11-21 | same, `lm_sweep.txt` | + CTC beam search (12) with 5-gram char LM from Coptic Scriptorium, α=0.5, β=1.0 (tuned on val) | **7.74** | 50.0 | — | −1.0 CER. Robust across α 0.2–0.5. Beam without LM gives no gain (8.76), so the gain is the LM. |
| E03 | 2026-11-21 | `runs/E03_resnet34_synth_only` | Same model, trained on synthetic lines only (`data/synth/v1`, 60k), validated on real val | 66.24 | 100.0 | — | Synthetic-only transfer is poor (synthetic loss 0.12, real CER 66). Fonts ≠ this scribe. Stopped at 7k steps. |
| E04 | 2026-11-21 | `runs/E04_resnet34_real_plus_synth50` | E02 + synthetic lines mixed in at 50% of each batch, 8k steps | 7.81 | 51.2 | — | −0.95 vs E02. Synthetic data works as a regularizer (train loss 0.08 vs 0.02). Confounded: 8k vs 6k steps, still improving at the end. |
| E04+LM | 2026-11-21 | same | + char LM (α=0.5, β=1.0 fixed from E02 sweep) | **7.27** | 45.9 | — | LM gain smaller on a stronger model (−0.5). |
| E05 | 2026-11-21 | `runs/E05_E04_12k_steps` | E04 recipe, 12k steps instead of 8k | 8.36 | 50.8 | — | Not better than E04 (7.81). Longer training doesn't help, and **seed noise is about ±0.5 CER**, so smaller differences aren't conclusive. |
| E06 | 2026-11-21 | `runs/E06_gold3x_pseudo1_synth30` | gold ×3 + 6,530 round-1 pseudo-labels (other scribes) + 30% synthetic, 12k steps | 7.99 | 47.9 | — | Neutral on SCAM val (same scribe), **but better on unseen hands** (OOD table below) |
| E07 | 2026-11-21 | `runs/E07_E04_height96` | E04 with 96 px input height instead of 64 (small marks: · ̇ ̈ are ~30% of errors) | 7.67 | 48.8 | — | Within noise of E04 on SCAM val. Slightly worse out of domain (22.08 vs 21.82) |
| E08 | 2026-11-21 | `runs/E08_gold3x_pseudo2a_synth30` | E06 recipe, but pool-1 pseudo-labels redone by E06 (round 2a: 9,682 lines kept vs 6,530) | 8.25 | 47.5 | — | OOD 20.39 vs 20.58 for E06: relabeling the same pages gives only a marginal gain |
| E09 | 2026-11-21 | `runs/E09_gold10x_pseudo3_pool12_synth30` | Round-3 pseudo-labels by E08 over pool 1 (442 pages) + pool 2 (700 more pages from 89 fascicles): 22,384 lines from 943 pages. gold ×10, synthetic 30%, 20k steps | 7.78 | 46.7 | — | + LM α=0.35 β=0.5: **7.30** (α=0.5 gave no gain; more confident models want a lower LM weight). OOD 20.09 |
| ens | 2026-11-21 | `scripts/recognize.py` (comma-separated ckpts) | Average CTC probabilities of E06+E08+E09 (± E04) | 8.68 / 7.52 | — | — | **Doesn't work**: CTC spikes of separately trained models land on different frames, so averaging smears them into blanks. Dropped |
| E10a | 2026-11-21 | `runs/E10a_E09_ft_gold` | Fine-tune E09 on gold only, lr 1e-4, 2k steps | 7.81 | 45.0 | — | + LM (α=0.35): **7.19**, best in-domain so far. OOD **20.03**, best so far; no forgetting |
| E11 | 2026-11-21 | `runs/E11_E10a_ft_printed1` | Fine-tune E10a (lr 2e-4, 4k steps) on printed Horner lines (round 1: 264 lines from 29 pages of John and 1 Cor, ×10) + gold ×3 + 8k pseudo + 20% synthetic | 8.18 | 45.0 | — | + LM: 7.70. **Printed 27.66 → 19.82.** OOD 20.17 (unchanged). Small in-domain cost |
| E12 | 2026-11-21 | `runs/E12_E10a_ft_printed2` | Like E11, but from E10a with round-2 printed lines (526 lines from 54 pages, re-read by E11), printed ×8, gold ×5, 5k steps (killed at 4.5k by a server restart; lr was already ~0) | 8.36 | 48.4 | 12.52 | + LM: val 7.74. **Printed 18.47** (best). OOD 20.19. **Shipped as the default** `models/recognizer.pt` |
| E13 | 2026-11-24 | `runs/E13_all_r34` | Cloud C1. E12's architecture, all data in one run from ImageNet (`data/mix/E13.lst`: gold ×10 + 22,384 pseudo + printed ×8; synthetic 30%), 20k steps | 8.54 | 49.2 | — | + LM 7.85. OOD 20.10, printed 19.05. One run matches the E09→E10a→E12 chain |
| E14 | 2026-11-24 | `runs/E14_all_r50_h512` | Like E13 but **ResNet50** (to layer3) + 2×BiLSTM-512, 21M params | 7.81 | 50.0 | — | + LM **7.45**. OOD 20.08, printed 18.94. Best all-rounder so far; see "Cloud round C1" |
| E13b | 2026-11-24 | `runs/E13b_ft_mix` | E13 + E12's mixed fine-tune (lr 2e-4, 4k steps) | 8.72 | 50.4 | — | + LM 8.28. No gain: the data is already in E13. Dropped |
| E14b | 2026-11-24 | `runs/E14b_ft_mix` | E14 + the same fine-tune | 8.03 | 47.9 | — | + LM 7.56, OOD 20.06, printed 18.87. Within noise of E14. Dropped |
| E15 | 2026-11-24 | `runs/E15_E14_seed1` | Cloud C2. E14 with seed 1 | 8.07 | 45.9 | — | + LM 7.63. OOD v2 10.99, printed 18.31. E14's two seeds: 7.45 / 7.63 |
| E16 | 2026-11-24 | `runs/E16_all_r101_h512` | Like E14 but **ResNet101** to layer3 (40M params) | 7.92 | 48.8 | — | + LM 7.74. OOD v2 11.00, printed 18.28. No gain over ResNet50, 1.7× slower. Dropped |
| E14c | 2026-11-24 | `runs/E14c_ft_gold` | E14 fine-tuned on gold only (E10a recipe: lr 1e-4, 2k steps) | **7.52** | 45.9 | — | + LM 7.38 (E10a: 7.19). OOD v2 11.01, printed 19.17. Best greedy so far |
| E17 | 2026-11-25 | `runs/E17_E14_showthrough` | Cloud C3. E14 recipe + **simulated show-through** (`--showthrough 0.3`: another line, mirrored, blurred, faded 25–60%, multiplied in) | 7.70 | 45.9 | — | + LM 7.30. OOD v2 10.95, printed 18.32 |
| E17s1 | 2026-11-25 | `runs/E17s1_E14_showthrough_seed1` | E17, seed 1 | **7.23** | 42.6 | **11.52** | + LM **6.98**, first under 7. OOD v2 10.99, printed **18.11** (best). **Shipped as the default** (replaces E12) |

## End-to-end page benchmark (photo → text)

Automatic segmentation, reading order and recognition on the 4 SCAM-A **val pages**. Page CER is the edit
distance between the predicted and gold letter streams of the whole page, ignoring line breaks
(`scripts/eval_pages.py`). Recognizer is E04 unless noted.

| # | Segmenter | Reading order | Greedy | + LM | Notes |
|---|---|---|---|---|---|
| P01 | Kraken default BLLA (fast box crops) | Kraken | 43.64 | 42.00 | Lines fragmented or merged across gaps; the generic model doesn't know this layout |
| P02 | **S01**: BLLA fine-tuned on 16 SCAM-A train pages (`runs/S01_blla_finetune_scam`, baselines derived from polygons) | Kraken | 18.51 | 17.67 | 3/4 pages at 6–11%. One page read as 4 blocks (L-top, R-top, L-bottom, R-bottom) |
| P03 | S01 | column-aware: cluster by line centre, read columns top to bottom | 26.84 | 25.89 | Fixed that page, broke another: tiny gutter detections chained the two columns |
| P04 | S01 | column-aware, specks filtered **before** ordering | 10.84 | **10.25** | Segmentation now adds only about 3 CER over gold line crops (7.27) |

## Out-of-domain benchmark: held-out biblical leaves, unseen scribes (`scripts/bible_ref.py`)

SCAM val is one scribe, but the goal is any Coptic hand. So: 30 Borg. copt. 109 fascicles **never used for
training or pseudo-labels** (233 pages), segmented with S01 and transcribed. Pages whose text matches the
Sahidic NT/OT (Coptic Scriptorium) are kept. Locator: 8-gram votes, ≥25% of the page's 8-grams in one
passage. Result: **35 biblical pages** (1 Timothy, Ephesians, Proverbs, Micah, Matthew…). Each gets a frozen
reference window (±250 letters). Score = letters-only infix edit distance / page length. Diacritics,
punctuation and spaces are ignored because edition and manuscript conventions differ. Spelling variants put
an unknown floor under this number (some pages reach 5–8%), but it's the same for every model.

**v2 (2026-11-24): the v1 number overcharged about 9 points.** Seven pages from fascicle X.32 are a **lectionary**:
mid-page they jump between books (Genesis 21 → Leviticus, Isaiah 13 → Daniel 10, Haggai 2 → Habakkuk 1). Others
run across a chapter boundary (Micah 1→2, Matthew 6→7, 1 Timothy 3→4). v1 compared every page with one window of
one chapter, so each jump counted as errors the model never made. Every model got almost the same score on those
pages, which is how it showed up. `bible_ref.py score2` (`score_segmented`) cuts the page's letters into
150-letter pieces, locates each piece, joins consecutive pieces from the same passage (a chapter plus its
neighbours), and scores each run against its own window. Checks:
- Pages that stay in one passage score exactly the same under v1 and v2 (25 of 35).
- Garbage doesn't benefit: shuffled lines 66.9 → 66.4, random letters 75.5 → 75.4.
- Printed (Horner, continuous text) barely moves, so the printed benchmark keeps v1.

| Model (all + LM) | SCAM val | OOD v1 | **OOD v2** | Notes |
|---|---|---|---|---|
| E02 real only | 8.76* | 24.72 | 16.15 | |
| E04 + synthetic | 7.81* | 21.82 | 13.01 | refs were located with this model's output (±250-letter margin keeps that fair) |
| E06 + pseudo round 1 | 7.99* | 20.58 | 11.61 | better or equal on 33/35 pages vs E04 |
| E07 height 96 | 7.67* | 22.08 | 13.32 | |
| E08 + pseudo round 2a (E06 relabels) | 8.25* | 20.39 | 11.39 | |
| E09 + pseudo round 3 (943 pages), LM α=0.35 | 7.78* | 20.09 | 11.01 | greedy 11.36 |
| E10a = E09 fine-tuned on gold | 7.81* | 20.03 | **10.98** | |
| E11 + printed round 1 | 8.18* | 20.17 | 11.17 | |
| E12 + printed round 2 (shipped) | 8.36* | 20.19 | 11.16 | |
| E13 all data at once, ResNet34 (cloud) | 8.50* | 20.10 | 11.05 | |
| E14 all data at once, ResNet50 + BiLSTM-512 (cloud) | 7.81* | 20.08 | 11.03 | |
| E14b E14 + mixed fine-tune (cloud) | 8.03* | 20.06 | **10.98** | |

Remaining v2 errors are spread out: the worst page is Acts 25 (XX.79_4r, 31.6), a short, damaged leaf.
Self-training is what moved this benchmark (16.2 → 11.0). Model size and the printed data didn't.

\* greedy val CER, from the main table.

## Test set (scored once, 2026-11-23, for the three finalists chosen on val)

SCAM-A test: 260 lines on 4 pages, never used for any decision before this. All with LM.

| Model | Val CER | **Test CER** | Test CER, spaces ignored | Test page CER (auto seg.) | OOD Bible | Printed |
|---|---|---|---|---|---|---|
| E04 (SCAM-A + synthetic only) | 7.27 | 11.99 | 8.67 | — | 21.82 | 29.86 |
| E10a (+ pseudo-labels, gold fine-tune) | 7.19 | **11.99** | **8.50** | **6.93** | **20.03** | 30.62 |
| E12 (+ aligned printed lines) | 7.92 | 12.52 | 9.02 | 7.35 | 20.19 | 18.47 |
| **E17s1** (2026-11-25: ResNet50, all data at once, show-through; cloud) | **6.98** | **11.52** | **8.01** | 7.04 | 20.12 (v2 10.99) | **18.11** |
| *paper: VAN (SCAM-A only)* | — | *7.50* | — | — | — | — |
| *paper: CRNN (SCAM-A only)* | — | *13.91* | — | — | — | — |

**Val→test gap.** Per page, the test CER of E10a is 6.9 and 8.1 on the XXV.120 pages and 13.5 and **20.0** on
the XXIX.159 pages. Page 0004 (1v) is **transcribed with word spaces**: 12.9% of its characters are spaces,
against 0–2.6% on every other SCAM-A page. The manuscript has no visible word division, so the models
(correctly) emit none, and each space is a deletion. That's where the "spaces ignored" column comes from.
On the official metric **we do not reach the paper's VAN (7.50)**; we sit between VAN and the paper's CRNN.
The page-level metric compares letter streams, so it isn't affected by the spaces.

**2026-11-25, E17s1** (the C3 finalist) was scored once on test. The page CERs above were re-run under identical
settings (LM α=0.35, β=0.5) for the comparison: E17s1 7.04, E10a 6.69, E12 7.52 on test pages; 8.87, 8.91, 9.2 on val
pages. E17s1 is best on test lines, val lines and printed, and equal on OOD. It's 0.35 worse than E10a on the 4 test
pages with auto segmentation, which is within noise. Caveat: E17s1 is the better of two seeds picked on val, which
flatters it slightly. Its sibling (seed 0, E17) has val 7.30. **E17s1 replaces E12 as the default** `models/recognizer.pt`.

## Printed benchmark: Horner's Sahidic NT, vol. 2 (Luke), 1911

Public-domain scan (archive.org `copticversionofn02hornuoft`). Each page is cropped to the main text above the
apparatus rule (`crop_above_rule.py`), segmented with S01, transcribed, located in the Sahidic NT and scored
the same way as the OOD benchmark. Recognition was poor enough that 8-gram locating found nothing, so pages
were located with 6-grams at ≥8% hits. That gave **14 pages, all Luke, in page order** (n41 → Luke 2, n421 → Luke 22).

| Model (all + LM) | Printed CER | Notes |
|---|---|---|
| E04 | 29.86 | No real printed training data. Horner's typeface (ϥ, ⲩ, ⲙ̅ⲙ) is far from both the synthetic fonts and the manuscripts |
| E06 | 27.56 | |
| E08 | 30.62 | |
| E09 | 27.66 | |
| E10a | 30.62 | |
| E11: + 264 aligned printed lines (John and 1 Cor; never Luke) | 19.82 | Alignment-based GT, round 1 |
| E12: + 526 aligned printed lines (round 2, bootstrapped with E11) | **18.47** | |

The Chaîne (1960) printed sample in the KELLIA repo reads much better qualitatively; its typeface is close to
New Athena / Antinoou. **Plan:** alignment-based ground truth from Horner vols 3 (John) and 4 (Epistles),
never Luke or the synoptic gospels, to train on real print (`align_printed_gt.py`).

## Pseudo-labelling (self-training on unlabeled Vatican pages)

- Source: 442 folio pages from 40 random fascicles of Borg. copt. 109 (White Monastery, many scribes), Vatican IIIF.
  **Both SCAM fascicles (XXV.120, XXIX.159) are excluded**, so no val/test page is ever seen, not even unlabeled.
- Segmented with S01: 21,470 line crops. Recognized with E04 + LM.
- Confidence filter calibrated on gold val lines (`pseudo_label.py calibrate`): keep lines whose lowest character
  peak probability is ≥ 0.8 **and** where greedy = LM decoding. On val that keeps 62% of lines at 4.0 CER (vs 7.3 for all).
- Round 1: 6,530 lines from 355 pages kept (min length 4). A visual spot-check showed mostly correct labels; typical
  errors are a missed decorated initial or a garbled hard line.

## Cloud round C1 (2026-11-24, Vast.ai RTX 3090)

Training moved off the laptop: it is node1's internet gateway, and an overheating crash on 2026-11-21 took both
machines offline (and killed E12). The box is a rented RTX 3090 (24 GB, 32 CPU cores, $0.18/h), instance label
`coptic-train`. Code and data are uploaded at the same absolute paths, so lists and scripts work unchanged.
Pipeline: [`runs/cloud_C1.sh`](runs/cloud_C1.sh). Scoring: `scripts/eval_suite.py` (val, unseen scribes and print in one pass).

Question: **is the model too small?** E09–E12 grew the data step by step on a 4 GB card. C1 trains on all of it at once
and compares two sizes on identical data:

| Run | Architecture | Data | Steps |
|---|---|---|---|
| E13 | current: ResNet34 to layer3, 2×BiLSTM-256 (11M params) | `data/mix/E13.lst`: gold ×10 + all 22,384 round-3 pseudo-labels + 526 aligned printed lines ×8; synthetic 30% | 20k from ImageNet |
| E14 | bigger: ResNet50 to layer3, 2×BiLSTM-512 (21M params) | same | 20k from ImageNet |
| E13b / E14b | each fine-tuned with E12's recipe (`E12.lst`, lr 2e-4) | gold ×5, printed ×8, 8k pseudo; synthetic 20% | 4k |

Check first: the shipped E12, re-scored on the cloud box, must give the local numbers (val+LM 7.74, OOD 20.19, printed 18.47).

**Results** (`eval_suite.py`, LM α=0.35 β=0.5; OOD and printed are letters CER vs. the edition):

| Model | Val greedy | Val + LM | Val + LM, no spaces | OOD (35 leaves) | Printed (14 pages) |
|---|---|---|---|---|---|
| E12 (shipped), re-scored on the box | 8.36 | 7.96 | 7.89 | 20.21 | 18.47 |
| E13: current size, all data at once | 8.50 | 7.85 | 7.78 | 20.10 | 19.05 |
| **E14: ResNet50 + BiLSTM-512** | **7.81** | **7.45** | **7.35** | 20.08 | 18.94 |
| E13b: E13 + mixed fine-tune | 8.72 | 8.28 | 8.15 | 20.21 | 19.04 |
| E14b: E14 + mixed fine-tune | 8.03 | 7.56 | 7.45 | 20.06 | 18.87 |

- **The box reproduces the laptop.** E12 re-scored gives printed 18.47 (identical) and OOD 20.21 (vs. 20.19). The val+LM
  7.96 vs. 7.74 is because 7.74 was the best point of an α/β sweep. All new rows use fixed α/β, so they compare fairly.
- **Bigger model: −0.4 val+LM over the same data** (E14 vs. E13), and +0.7 greedy. OOD and printed are unchanged.
  The gain is within the ±0.5 seed noise seen before, so C2 repeats E14 with another seed.
- **One run replaces the chain.** E13 alone matches E12 everywhere (val +LM 7.85 vs. 7.96, OOD 20.10, printed 19.05 vs. 18.47).
- **OOD is stuck at about 20** for every model since E09. The per-page analysis found why: part of that 20 was the
  benchmark, not the models (lectionary pages, see "OOD v2" below). With v2 it's about 11.0 for all of them.
- Cost: both 20k-step runs together took 41 min on the 3090 (the laptop took 2.5 h for one). C1 total, including upload: about $0.40.

## Cloud round C2 (2026-11-24): noise, more size, gold fine-tune

Pipeline: [`runs/cloud_C2.sh`](runs/cloud_C2.sh). Same box. Three jobs at once, 70 min. E10a was re-scored with the same suite as a reference.

| Model | Val greedy | Val + LM | OOD v2 | Printed |
|---|---|---|---|---|
| E10a (manuscript specialist, laptop) | 7.81 | **7.19** | **10.98** | 30.62 |
| E12 (shipped default, laptop) | 8.36 | 7.96 | 11.18 | 18.47 |
| E14 ResNet50, seed 0 | 7.81 | 7.45 | 11.03 | 18.94 |
| E15 ResNet50, seed 1 | 8.07 | 7.63 | 10.99 | 18.31 |
| E16 ResNet101 | 7.92 | 7.74 | 11.00 | **18.28** |
| E14c E14 + gold fine-tune | **7.52** | 7.38 | 11.01 | 19.17 |

- **The bigger model is worth about 0.3** (ResNet50 seeds average 7.54 vs. E13's 7.85 with LM). ResNet101 adds nothing.
- **OOD and printed are flat** across every model since E09: 11.0 ± 0.05 and 18.3–19.2. Neither size nor recipe moves them.
- **Val is noisy.** 242 lines; 0.5 CER is about 14 graphemes. E14c's remaining 207 greedy errors: 30% letter
  substitutions, 53% missing or extra letters, 15% wrong marks, 1.4% spaces. **10 lines (4% of val) carry 30% of
  the errors**: ink bleeding through from the other side, a torn edge, an enlarged initial with marginal notes.
  That's the target of C3.
- Nothing here clearly beats the shipped pair (E12 default, E10a manuscript), so the test set stays untouched and
  nothing new is shipped yet.

## Cloud round C3 (2026-11-25): simulated bleed-through

Pipeline: [`runs/cloud_C3.sh`](runs/cloud_C3.sh). On 30% of real training lines, `LineDataset._ghost` blends in another
training line, mirrored, blurred (σ 0.5–2), shifted up or down, and faded to 25–60% (ink on the other side of the
leaf showing through). Same recipe and seeds as E14/E15 otherwise.

| | Val greedy | Val + LM | OOD v2 | Printed |
|---|---|---|---|---|
| E14 / E15 (no show-through, seeds 0/1) | 7.81 / 8.07 | 7.45 / 7.63 | 11.03 / 10.99 | 18.94 / 18.31 |
| **E17 / E17s1 (show-through, seeds 0/1)** | 7.70 / **7.23** | 7.30 / **6.98** | 10.95 / 10.99 | 18.32 / **18.11** |

- **Both augmented seeds beat both plain seeds**: −0.4 val+LM on average (7.14 vs. 7.54), −0.5 greedy. It's the first
  change since E09 that clearly clears seed noise. OOD and printed don't get worse (printed is slightly better).
- E17s1 was scored on test (see "Test set") and shipped.
- Cost of C1–C3: about 4.5 GPU-hours on the rented 3090, about $1.50 including upload and idle time. The box was destroyed afterwards.

## Notes

### Leakage check: SCAM vs Coptic Scriptorium (2026-11-21)
The synthetic text and the character LM come from Coptic Scriptorium, so I checked that SCAM-A's text isn't in it.
- No 3-line window from val (76 windows) or test (85 windows) appears verbatim. 2 of 303 train windows do (Helias martyrdom).
- 20-letter windows (letters only) found in Coptic Scriptorium: train 1.0%, val 0.8%, test 0.7%. That's consistent with stock phrases, not a shared text.
- SCAM-A's text is church history, with names like Hosius of Cordova. Character LM perplexity on SCAM val is about 15 (letters only), against about 6 on held-out Coptic Scriptorium. Real domain shift, but not leakage.

### Kraken quirks found
- `ketos train --log-dir` must already exist.
- In 7.2, `return_logits` returns decoded tuples for bbox lines; per-frame probabilities are on `model.net.outputs`.
- Checkpoints are 47 MB per epoch and all are kept.
