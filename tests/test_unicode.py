"""Unit tests for Unicode cleaning (P0)."""
from __future__ import annotations

from app.cleaner.unicode import (
    clean_text,
    indic_script_ratio,
    is_embeddable,
    normalize_unicode,
    quality_score,
)


def test_normalize_unicode_nfc():
    raw = "অ\u09BE"  # decomposed vowel sign
    assert normalize_unicode(raw) == "অা"


def test_indic_script_ratio_assamese():
    text = "অসমীয়া ভাষা"
    assert indic_script_ratio(text) > 0.8


def test_quality_score_rejects_garbage():
    text = "Łłʢˤ͋͌ garbage"
    assert quality_score(text) < 0.45


def test_clean_text_preserves_qa_lines():
    text = "প্ৰশ্ন?\nউত্তৰ"
    cleaned, q = clean_text(text, preserve_line_breaks=True)
    assert "?" in cleaned
    assert "\n" in cleaned
    assert q > 0.3


def test_is_embeddable_threshold():
    good = "অসমীয়া প্ৰশ্নোত্তৰ ডাটাছেট"
    assert is_embeddable(good, min_quality=0.2)
    assert not is_embeddable("x", min_quality=0.45)
