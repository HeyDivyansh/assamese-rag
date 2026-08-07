"""Header/footer and page-number stripping (P1)."""
from __future__ import annotations

import re
from collections import Counter

from app.pipeline.types import ParsedPage

_PAGE_NUM = re.compile(
    r"^(?:পৃষ্ঠা|পৃ\.|Page|P\.?)\s*[\d০-৯]+[\s./-]*$",
    re.IGNORECASE,
)
_ONLY_NUM = re.compile(r"^[\d০-৯./\s-]{1,12}$")


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower())


def detect_repeated_lines(pages: list[ParsedPage], *, min_pages: int = 3) -> set[str]:
    """Lines repeated on >=40% of pages are likely headers/footers."""
    if len(pages) < min_pages:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        lines = {_normalize_line(ln) for ln in page.text.split("\n") if ln.strip()}
        for ln in lines:
            if 3 <= len(ln) <= 120:
                counts[ln] += 1
    threshold = max(2, int(len(pages) * 0.4))
    return {ln for ln, c in counts.items() if c >= threshold}


def strip_headers_footers(pages: list[ParsedPage]) -> list[ParsedPage]:
    repeated = detect_repeated_lines(pages)
    cleaned: list[ParsedPage] = []
    for page in pages:
        kept: list[str] = []
        for ln in page.text.split("\n"):
            norm = _normalize_line(ln)
            if norm in repeated:
                continue
            if _PAGE_NUM.match(ln.strip()) or _ONLY_NUM.match(ln.strip()):
                continue
            kept.append(ln)
        new_text = "\n".join(kept).strip()
        cleaned.append(
            ParsedPage(
                page_number=page.page_number,
                text=new_text,
                confidence=page.confidence,
                engine=page.engine,
                blocks=page.blocks,
                indic_ratio=page.indic_ratio,
            )
        )
    return cleaned
