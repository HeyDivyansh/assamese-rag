"""Unicode normalization and OCR corruption repair (P0)."""
from __future__ import annotations

import re
import unicodedata

# Control + invisible characters.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INVISIBLE = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\ufffe\uffff]"
)
# Indic + Latin (mixed docs).
_INDIC = re.compile(r"[\u0980-\u09FF]")
_LATIN = re.compile(r"[A-Za-z]")
# Common OCR garbage symbols.
_GARBAGE = re.compile(r"[Łłʢˤ͍͎͓͔͕͖͙͚͋͌͐͑͒͗͛͘͜͟͝͞͠͡]")
# Known OCR confusions (Latin/garbage → Bengali-Assamese approximations).
_OCR_REPAIRS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Ł"), "ল"),
    (re.compile(r"ł"), "ল"),
    (re.compile(r"ʢ"), ""),
    (re.compile(r"ˤ"), ""),
    (re.compile(r"͋|͌|͍|͎|͐|͑|͒|͓|͔|͕|͖|͗|͘|͙|͚|͛|͜|͝|͞|͟|͠|͡"), ""),
]
_HYPHEN_WRAP = re.compile(r"(\w)[-\u00ad]\n(\w)")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK = re.compile(r"\n{3,}")


def indic_script_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return len(_INDIC.findall(text)) / len(chars)


def garbage_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_GARBAGE.findall(text)) / max(len(text), 1)


def quality_score(text: str) -> float:
    """0..1 quality heuristic for embed gating."""
    if not text or len(text.strip()) < 5:
        return 0.0
    indic = indic_script_ratio(text)
    garbage = garbage_char_ratio(text)
    latin = len(_LATIN.findall(text)) / max(len(text.replace(" ", "")), 1)
    # Mixed Assamese-English is OK; heavy garbage is not.
    score = 0.5 * indic + 0.3 * (1.0 - garbage * 10) + 0.2 * min(latin + indic, 1.0)
    return max(0.0, min(1.0, score))


def normalize_unicode(text: str) -> str:
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub("", text)
    text = _INVISIBLE.sub("", text)
    text = _GARBAGE.sub("", text)
    for pattern, repl in _OCR_REPAIRS:
        text = pattern.sub(repl, text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_WRAP.sub(r"\1\2", text)
    return text


def clean_text(
    text: str,
    *,
    preserve_line_breaks: bool = False,
) -> tuple[str, float]:
    """Normalize text and return (cleaned, quality_score)."""
    text = normalize_unicode(text)
    if preserve_line_breaks:
        lines = [_MULTISPACE.sub(" ", ln).strip() for ln in text.split("\n")]
        text = "\n".join(ln for ln in lines if ln)
    else:
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = _MULTI_BLANK.sub("\n\n", text)
        text = _MULTISPACE.sub(" ", text)
    text = text.strip()
    return text, quality_score(text)


def is_embeddable(text: str, min_quality: float) -> bool:
    _, q = clean_text(text)
    return bool(text.strip()) and q >= min_quality and len(text.strip()) >= 10
