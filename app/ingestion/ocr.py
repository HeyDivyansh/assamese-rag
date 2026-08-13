"""Pluggable OCR layer.

PaddleOCR is the primary engine, but its Assamese/Bengali-script accuracy is
UNVERIFIED (spec §2). So:
  * every page gets a confidence score,
  * pages below OCR_CONFIDENCE_THRESHOLD are re-run through the fallback engine
    (Sarvam Vision by default),
  * the engine used + confidence is recorded per page so weak pages are flagged.

Engines implement a tiny interface (`OcrEngine`) making them swappable via the
OCR_PRIMARY_ENGINE / OCR_FALLBACK_ENGINE env vars.
"""
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.cleaner.unicode import indic_script_ratio, latin_script_ratio
from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

# PaddleOCR 2.9+ removed per-language codes like ``bn``; Indic OCR uses ``devanagari``.
_PADDLE_LANGS = frozenset({
    "ch", "en", "korean", "japan", "chinese_cht", "ta", "te", "ka",
    "latin", "arabic", "cyrillic", "devanagari",
})


@dataclass
class PageOCR:
    page_number: int  # 1-indexed
    text: str
    confidence: float  # 0..1 average over detected lines
    engine: str


@dataclass
class OCRResult:
    pages: list[PageOCR] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def avg_confidence(self) -> float:
        if not self.pages:
            return 0.0
        return sum(p.confidence for p in self.pages) / len(self.pages)


def extract_pdf_text_pages(pdf_bytes: bytes) -> list[PageOCR]:
    """Extract embedded PDF text (digital PDFs) without OCR."""
    import pypdfium2 as pdfium

    pages: list[PageOCR] = []
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            text = (textpage.get_text_range() or "").strip()
            pages.append(PageOCR(i + 1, text, 0.0, "pdf_text_layer"))
    finally:
        pdf.close()
    return pages


def _should_use_text_layer(text: str) -> bool:
    """Use native PDF text when it looks like real content, not empty."""
    stripped = text.strip()
    if len(stripped) < 20:
        return False
    return (
        indic_script_ratio(stripped) >= 0.20
        or latin_script_ratio(stripped) >= 0.20
    )


def _page_needs_fallback(page: PageOCR) -> bool:
    """Confidence alone is not enough — garbled OCR can still score high."""
    if page.confidence < settings.ocr_confidence_threshold:
        return True
    if len(page.text.strip()) > 40:
        indic = indic_script_ratio(page.text)
        latin = latin_script_ratio(page.text)
        if indic < 0.15 and latin < 0.15:
            return True
    return False


def _paddle_lang_for_page(
    native_text: str | None,
    *,
    document_hint: str | None = None,
) -> str:
    """Pick PaddleOCR lang from page text, then whole-document hint."""
    for text in (native_text, document_hint):
        stripped = (text or "").strip()
        if len(stripped) < 20:
            continue
        latin = latin_script_ratio(stripped)
        indic = indic_script_ratio(stripped)
        if latin >= 0.20 and latin >= indic:
            return settings.ocr_paddle_lang_latin
        if indic >= 0.15:
            return settings.ocr_paddle_lang_indic
    return settings.ocr_paddle_lang_indic


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> list["PIL.Image.Image"]:  # noqa: F821
    """Render each PDF page to a PIL image using pypdfium2."""
    import pypdfium2 as pdfium

    images = []
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        scale = dpi / 72.0
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            images.append(bitmap.to_pil())
    finally:
        pdf.close()
    return images


class OcrEngine(ABC):
    name: str

    @abstractmethod
    def ocr_image(self, image, page_number: int) -> PageOCR: ...


class PaddleOcrEngine(OcrEngine):
    name = "paddleocr"

    def __init__(self, lang: str | None = None) -> None:
        lang = lang or settings.ocr_paddle_lang_indic
        if lang not in _PADDLE_LANGS:
            raise ValueError(
                f"Unsupported PaddleOCR lang {lang!r}; "
                f"choose one of {sorted(_PADDLE_LANGS)}"
            )
        self._lang = lang
        self._ocr = None

    def _get(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR

            # Assamese uses Eastern Nagari script; PaddleOCR 2.9+ maps Indic OCR
            # to ``devanagari`` (``bn`` was removed).
            self._ocr = PaddleOCR(use_angle_cls=True, lang=self._lang)
        return self._ocr

    def ocr_image(self, image, page_number: int) -> PageOCR:
        import numpy as np

        ocr = self._get()
        arr = np.array(image.convert("RGB"))
        result = ocr.ocr(arr, cls=True)
        lines: list[str] = []
        confs: list[float] = []
        for block in result or []:
            for line in block or []:
                # line = [box, (text, score)]
                try:
                    text, score = line[1]
                    lines.append(text)
                    confs.append(float(score))
                except Exception:  # noqa: BLE001
                    continue
        conf = sum(confs) / len(confs) if confs else 0.0
        return PageOCR(page_number, "\n".join(lines), conf, self.name)


class SarvamVisionEngine(OcrEngine):
    """Sarvam document-intelligence / Vision OCR fallback."""

    name = "sarvam_vision"

    def ocr_image(self, image, page_number: int) -> PageOCR:
        from app.llm.sarvam_client import sarvam_vision_ocr

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        text, conf = sarvam_vision_ocr(buf.getvalue(), page_number=page_number)
        return PageOCR(page_number, text, conf, self.name)


class NullEngine(OcrEngine):
    name = "none"

    def ocr_image(self, image, page_number: int) -> PageOCR:
        return PageOCR(page_number, "", 0.0, self.name)


_ENGINES = {
    "paddleocr": PaddleOcrEngine,
    "sarvam_vision": SarvamVisionEngine,
    "none": NullEngine,
}

_paddle_by_lang: dict[str, PaddleOcrEngine] = {}


def _make_engine(name: str) -> OcrEngine:
    cls = _ENGINES.get(name, NullEngine)
    return cls()


def _get_paddle_engine(lang: str) -> PaddleOcrEngine:
    if lang not in _paddle_by_lang:
        _paddle_by_lang[lang] = PaddleOcrEngine(lang=lang)
    return _paddle_by_lang[lang]


def run_ocr(pdf_bytes: bytes) -> OCRResult:
    """Extract text page-by-page: PDF text layer first, then OCR + fallback."""
    primary = _make_engine(settings.ocr_primary_engine)
    fallback_name = settings.ocr_fallback_engine
    fallback = _make_engine(fallback_name) if fallback_name != "none" else None

    native_pages = extract_pdf_text_pages(pdf_bytes)
    document_hint = " ".join(p.text for p in native_pages if p.text.strip())
    images = pdf_to_images(pdf_bytes)
    result = OCRResult()
    for idx, image in enumerate(images, start=1):
        native = native_pages[idx - 1] if idx <= len(native_pages) else None
        if native and _should_use_text_layer(native.text):
            log.info(
                "ocr.used_pdf_text_layer",
                page=idx,
                chars=len(native.text),
                indic_ratio=round(indic_script_ratio(native.text), 3),
            )
            result.pages.append(
                PageOCR(idx, native.text, 0.98, "pdf_text_layer")
            )
            continue

        if settings.ocr_primary_engine == "paddleocr":
            paddle_lang = _paddle_lang_for_page(
                native.text if native else None,
                document_hint=document_hint,
            )
            page = _get_paddle_engine(paddle_lang).ocr_image(image, idx)
        else:
            page = primary.ocr_image(image, idx)
        if fallback is not None and _page_needs_fallback(page):
            log.warning(
                "ocr.low_confidence_fallback",
                page=idx,
                primary=primary.name,
                confidence=round(page.confidence, 3),
                indic_ratio=round(indic_script_ratio(page.text), 3),
                fallback=fallback.name,
            )
            fb = fallback.ocr_image(image, idx)
            # Prefer fallback when it has more script signal or higher confidence.
            page_script = max(
                indic_script_ratio(page.text), latin_script_ratio(page.text)
            )
            fb_script = max(
                indic_script_ratio(fb.text), latin_script_ratio(fb.text)
            )
            if fb_script > page_script or fb.confidence >= page.confidence:
                page = fb
        log.info(
            "ocr.page_done",
            page=idx,
            engine=page.engine,
            confidence=round(page.confidence, 3),
            indic_ratio=round(indic_script_ratio(page.text), 3),
            chars=len(page.text),
        )
        result.pages.append(page)
    return result
