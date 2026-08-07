"""Text cleaning for OCR output.

  * NFC Unicode normalization (critical for Assamese/Bengali conjuncts).
  * Strip common OCR artifacts / control chars.
  * Repair words broken across line wraps (hyphen + newline, and bare newlines
    inside a paragraph).
  * Preserve line breaks for Q&A-style documents (question on one line, answer on next).
"""
from __future__ import annotations

import re
import unicodedata

# Control chars except tab/newline.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Hyphenated line break: "exam-\nple" -> "example"
_HYPHEN_WRAP = re.compile(r"(\w)[-\u00ad]\n(\w)")
# 3+ blank lines collapse to a paragraph break.
_MULTI_BLANK = re.compile(r"\n{3,}")
# Repeated spaces.
_MULTISPACE = re.compile(r"[ \t]{2,}")


def looks_like_qa_lines(text: str) -> bool:
    """Detect Q&A datasets: multiple lines ending with a question mark."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    q_lines = sum(1 for ln in lines if ln.endswith("?") or ln.endswith("？"))
    return q_lines >= 2


def clean_text(text: str, *, preserve_line_breaks: bool | None = None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_WRAP.sub(r"\1\2", text)

    if preserve_line_breaks is None:
        preserve_line_breaks = looks_like_qa_lines(text)

    if preserve_line_breaks:
        lines = [_MULTISPACE.sub(" ", ln).strip() for ln in text.split("\n")]
        text = "\n".join(ln for ln in lines if ln)
    else:
        # Join a single hard newline inside a paragraph (line-wrap), but preserve
        # paragraph breaks (blank line).
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = _MULTI_BLANK.sub("\n\n", text)
        text = _MULTISPACE.sub(" ", text)
    return text.strip()


def clean_pages(pages: list[tuple[int, str, float]]) -> list[tuple[int, str, float]]:
    """Clean a list of (page_number, text, confidence) tuples."""
    qa_doc = any(looks_like_qa_lines(txt) for _, txt, _ in pages)
    return [
        (pn, clean_text(txt, preserve_line_breaks=qa_doc), conf)
        for pn, txt, conf in pages
    ]
