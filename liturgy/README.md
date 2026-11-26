# Liturgy sing-along: text and audio synced word by word

The goal is to take a recording of a Coptic hymn or liturgy part (from YouTube, a cantor, or a class) and the text in Coptic, English letters, English and Arabic. The output is a page where each word lights up as it's sung. Students can click any word to jump to it, repeat a line, or slow the recording down.

```bash
.venv/bin/python liturgy/scripts/tasbeha.py lessons        # 75 hand-timed tasbeha lessons + mp3s (answer key)
.venv/bin/python liturgy/scripts/tasbeha.py texts          # all ~2,000 tasbeha texts (Coptic, English letters, English, Arabic)
.venv/bin/python liturgy/scripts/make_lesson.py --text 32 --audio youtube:3eFEgD7Jg-w   # -> liturgy/lessons/*.html
.venv/bin/python liturgy/scripts/align.py eval --garbage=-2 --name A02_g2_lines           # score against the answer key
```

## How it works

1. **Text.** It comes from the [tasbeha.org hymn library](https://tasbeha.org/hymn_library/): about 2,000 texts. Every Coptic word is paired with its English-letter transliteration, and every line has English and Arabic.
2. **Alignment.**
   - **Model:** MMS-300m forced aligner ([`MahmoudAshraf/mms-300m-1130-forced-aligner`](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner)), a wav2vec2 model trained on 1,130 languages. Coptic isn't one of them, so we feed it tasbeha's transliteration, which spells out the Greco-Bohairic pronunciation.
   - **Search:** a CTC Viterbi search gives every letter a 20 ms frame, and a word starts at its first letter.
   - **Extra states:** two additions handle audio that doesn't match the text:
     - a "skip audio" state soaks up introductions, spoken Arabic and other singers' parts;
     - a "skip line" option lets the aligner leave out a whole line that isn't sung. It costs a set penalty per letter.
   - **Speaker labels:** lines such as "Deacon:" or "People:" are shown but not aligned.
3. **Player.** `player/player.html` is one self-contained page per lesson with the audio embedded, so it works offline. It has:
   - columns for Coptic, English letters, English and Arabic;
   - repeat line, 50/75/100% speed (pitch kept) and click-to-seek;
   - a "Fix timings" mode: a teacher selects a word, presses T as it's sung, then downloads the corrected JSON.

## Answer key and results

tasbeha.org has 75 "lessons", recordings with hand-checked word timings. Our timings agree with theirs to within 20 ms on most words, which means tasbeha built theirs with a similar aligner and then fixed it by hand. So the remaining disagreements show where automatic alignment needs a human.

| Run | Setup | Words ≤0.25 s | ≤0.5 s | ≤1 s | Line starts ≤0.5 s |
|---|---|---|---|---|---|
| A01 | plain forced alignment, recorded lines only | 84.7 | 86.3 | 88.2 | 83.0 |
| A02 | + skip-audio state before each line (−2/frame; −1 to −10 all within 0.1) | **85.4** | **87.0** | **88.6** | **83.9** |
| A03 | skip-audio only at start/end | 84.7 | 86.3 | 88.2 | 83.0 |
| A04 | A02, but given the **whole** text (incl. lines not in the recording) | 85.2 | 86.8 | 88.2 | 82.7 |
| A05 | A04 + skip-line −9/letter (−4 to −13 within 0.8; −2 drops real lines) | 85.2 | 86.8 | 88.2 | 82.5 |

### Round 2: drawn-out hymns (all with the whole text and skip-line −9/letter)

"Drawn-out" means 24 lessons where some word is held at least 15 s (1,338 words); the other 51 lessons have 1,519 words. "After a long hold" means words that follow a note held at least 8 s.

| Run | Change | All ≤0.5 s | Drawn-out | Others | After a long hold |
|---|---|---|---|---|---|
| A07 bp0 | as A05 | 86.8 | 86.5 | 87.0 | 75.5 |
| A07 bp2 | blank −2 (= skip-audio −2) | 92.0 | | | 73.2 |
| A08 | blank and skip-audio *not* equal (1.5/2, 2/3, 2.5/3) | ≈ A07 bp0 | | | |
| A08 | skip-audio cheaper than blank (2.5/2, 2/1.5) | 17 (collapses) | | | |
| A09 | blank = skip-audio, 0.5 to 3 | 92.0–92.2 | 87.5 | 96.1 | 73.2–73.6 |
| A10 | + held vowels (a letter may come back after a pause) | 92.1 | 87.5 | 96.1 | 74.1 |
| **A11** | **skip-audio −1.02, blank −1, held vowels (default)** | **92.7** | **88.7** | **96.2** | |
| A11 | skip-audio −1.05 / −1.1 / −1.4 / −1.7 | 91.3 / 89.6 / 88.2 / 87.4 | 88.6–86.8 | | |

So skipping the audio between lines must cost about the same as a pause, and just slightly more.

**Why held notes are hard.** During a 30–70 s melisma, the model hears a plain vowel ("a … e … a") that doesn't follow the written word.
- **Apekran:** "pekran" is held 65 s, and its "a" sound runs straight into "ernishti". The aligner can only find "ernishti" at its consonants ("-shti", about 46 s after tasbeha's start). It also skips the whole 160 s opening melisma as if it weren't in the text.
- **Sotees Amen:** the model hears "a" for 0–124 s, all the way through "Sothis" and "ameen".
- **The fix:** fine-tune the acoustic model on hand-timed chant, starting with tasbeha's 75 lessons plus teacher corrections, so it learns how cantors carry a word through a melisma.

### Round 3: fine-tuning the acoustic model on hand-timed chant (rented RTX 3090, about $0.20)

**Data and split** (`ft_data.py`):
- The 75 lessons are split by hymn. Train is 53 recordings (108 min of timed chant), val is 6 recordings, and test is 16 recordings of 13 hymns. Test always includes the hardest hymns: Apekran, Sotees Amen, both Shere Maria recordings, Tai Shori and Yes Pateer.
- A segment is a run of words from tasbeha's word starts (up to 25 s), or one held word (up to 60 s), with its letters as the CTC target.
- Spans with no letters, like the patriarch's name, are left out.

**Training** (`ft_train.py`):
- The CNN feature encoder is frozen; SpecAugment uses p=0.05; batches hold 120 s of audio in bf16, for 1,500 steps.
- The full alignment pipeline runs on val every 100 steps, and the best checkpoint is kept.

| Model | Test words ≤0.5 s | ≤1 s | After a long hold (n=63) | Apekran | Sotees | Shere Maria | Tai Shori | Yes Pateer |
|---|---|---|---|---|---|---|---|---|
| base (A11 settings) | 77.3 | 79.7 | 50.8 | 58 | 67 | 78 | 81 | 78 |
| F01 lr 2e-5 (bad mask init, see below) | 74.3 | 77.1 | 50.8 | 68 | 67 | 96 | 81 | 69 |
| F02 lr 5e-5 | 76.2 | 78.5 | **63.5** | **70** | 67 | 91 | **94** | 76 |
| F03 lr 3e-5, lower 12 layers frozen | 76.2 | 78.3 | 60.3 | 68 | 67 | 96 | 88 | 73 |
| **F03 × 0.5 + base × 0.5** (chosen on val, test scored once) | **77.6** | 79.4 | 55.6 | 65 | 67 | 83 | 88 | 74 |

**What it shows:**
- Fine-tuning does learn melismas: words after a long hold go from 51% to 64%, Apekran from 58% to 70%, and Tai Shori from 81% to 94%.
- On its own it loses just as much elsewhere. Most of the loss is Litany of the Fathers (27 → 18 words), where the cantor names a different pope than the text.
- The 50/50 mix with the base model keeps most of the gain and loses nothing overall. It is the default in `make_lesson.py` (`--ft-model`, `--mix`).
- With 108 minutes of training chant, a 13-hymn test set and val already at 93–95%, these differences are a few words each. More timed chant is the way forward, for example teacher corrections from the player.

**Gotchas:**
- `masked_spec_embed` loads uninitialised from this checkpoint. Garbage values there gave NaN losses (F01's early collapse, and F02's first attempt). `ft_train.py` now initialises it.
- transformers 5 requires torch ≥ 2.6; the Vast image has torch 2.5.1, so it uses transformers 4.x.

### Round 4: following the melody instead of the letters (`ref_align.py pairs`)

**Idea:** carry word timings from a hand-timed recording of a hymn to another recording of the same hymn. DTW matches the two recordings every 100 ms on two features:
- pitch, in semitones relative to each recording's own median (pYIN), so key doesn't matter;
- the speech model's letter probabilities.

**Test:** 11 hymns that tasbeha has as two hand-timed recordings, transferred both ways, 716 words.

| Method | ≤0.5 s | ≤1 s | ≤2 s |
|---|---|---|---|
| letter aligner (A11) | **93.4** | **94.3** | **95.4** |
| DTW, sound only | 85.9 | 89.5 | 91.2 |
| DTW, pitch only | 45.4 | 53.5 | 64.2 |
| DTW, sound + pitch | 76.3 | 83.0 | 86.3 |

Restricting both recordings to the sung span, the best weighting (sound + 0.3 pitch) gives 83.0 on 7 pairs.

**What it shows:**
- Melody transfer works only when both recordings use the same tune, as with Pi-nishti (95%).
- **The two Shere Maria recordings are different tunes of the same text.** In 8021, "Shere" is held 180 s and "ne" 122 s, and the rest goes quickly. In 7837, the long melismas fall on ethnesos (31 s), thee-etasmisi (33 s), emefnouti (32 s) and Pekiot (28 s). A single template per hymn can't cover that.
- Pitch on its own is weak: ornaments vary between cantors, and church reverb makes pitch tracking noisy.

**Possible uses later:**
- a tune library with one timed reference per tune, used only when the DTW match is confident;
- a soft prior on relative word durations from a matching reference.

Comparison pages for checking by ear: `compare_page.py <lesson ids>` writes `lessons/compare_*.html`.

Skipping lines doesn't change the score. It's on by default anyway, because without it an unsung line, such as the Creed's opening deacon call, gets squeezed into the first seconds and highlighted wrongly.

### Where it fails (the ~13% of words off by more than 0.5 s)

- **Audio that isn't in the text:**
  - the current pope's or bishop's name said where the text has a placeholder;
  - optional words printed in parentheses;
  - a response repeated;
  - spoken Arabic introductions ("في القداس… يقول الشماس").
- **Very drawn-out hymns** (the "Great" tunes, 10–16 minutes for about 120 words): the model hears the long vowels as silence, so it loses its place.
- **Word starts in a breath or pause:** the first letter sometimes starts 1–2 s early.

Three different YouTube cantors singing Tai Shori produce very consistent timings, for example ennoub at 13.2, 13.8 and 13.8 s, and pi-oweeb at 90, 85 and 93 s. That's a good sign, since the tune is fixed.

## Ideas to try next

- Treat words in parentheses as optional.
- Add a placeholder for the patriarch's or bishop's name.
- Add the Arabic text as alternative audio, for recordings that sing a line in Arabic too.
- Handle long vowels in melismatic hymns: give a vowel's frames to its word, and set onset from the energy rise.
- Align whole liturgies from YouTube, which run 1–2 hours and mix Coptic, Arabic and English.
- Teacher corrections will become new answer-key data.
