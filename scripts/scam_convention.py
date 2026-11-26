"""Convert Coptic Scriptorium diplomatic text to SCAM transcription conventions.

SCAM conventions (from inspecting page_annotations_RO.xml):
  - single-letter supralinear stroke  -> U+0307 COMBINING DOT ABOVE
  - multi-letter stroke (nomina sacra) -> U+FE26 on every covered letter
  - diaeresis U+0308 kept, middle dot · kept, '.' marks illegible letters
CS uses U+0304/U+0305 for strokes and U+FE24..U+FE25 half-macron spans.
Editorial marks that are not visible on the page are removed.
"""
import unicodedata

STROKE = "̇"
NS = "︦"
DROP = set("[](){}⟨⟩⟦⟧‹›❮❯`⸗̣̀̂́᷍ͅ⸗")
MAP = {"̄": STROKE, "̅": STROKE, "⸱": "·", "‧": "·", "˽": ""}

# Characters SCAM-A uses (plus a few Coptic letters/uppercase that may appear).
ALLOWED = set(
    "ⲁⲃⲅⲇⲉⲋⲍⲏⲑⲓⲕⲗⲙⲛⲝⲟⲡⲣⲥⲧⲩⲫⲭⲯⲱϣϥϧϩϫϭϯⳁ"
    "·.: …" + STROKE + NS + "̈"
)


def to_scam(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    out = []
    in_span = False
    for ch in s:
        if ch in DROP:
            continue
        ch = MAP.get(ch, ch)
        if not ch:
            continue
        if ch == "︤":  # span opens on the previous letter
            in_span = True
            out.append(NS)
            continue
        if ch == "︥":  # span closes on the previous letter
            in_span = False
            out.append(NS)
            continue
        if unicodedata.combining(ch):
            if out and out[-1] == ch:
                continue
            out.append(ch)
            continue
        if in_span and out and not unicodedata.combining(out[-1]):
            out.append(NS)  # letter strictly inside the span gets the mark too
        out.append(ch.lower())
    s = unicodedata.normalize("NFC", "".join(out))
    # dedupe repeated FE26 on one base letter
    while NS + NS in s:
        s = s.replace(NS + NS, NS)
    return " ".join(s.split())


def is_allowed(s: str) -> bool:
    return bool(s) and all(ch in ALLOWED for ch in unicodedata.normalize("NFD", s))
