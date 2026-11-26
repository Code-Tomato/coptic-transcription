"""One text normalization used everywhere (ground truth, synthetic text, predictions).

Every model is scored after the same normalization, so changing this file
changes the benchmark: bump BENCHMARK_VERSION in eval.py if you edit it.
"""
import re
import unicodedata

_WS = re.compile(r"\s+")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace(" ", " ").replace("\t", " ")
    s = _WS.sub(" ", s).strip()
    return s
