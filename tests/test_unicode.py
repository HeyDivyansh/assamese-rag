"""Unit tests for Unicode cleaning (P0)."""
from __future__ import annotations

from app.cleaner.unicode import (
    clean_text,
    detect_language,
    detect_query_language,
    indic_script_ratio,
    is_embeddable,
    latin_script_ratio,
    normalize_unicode,
    quality_score,
)


def test_normalize_unicode_nfc():
    raw = "অ\u09BE"  # decomposed vowel sign
    assert normalize_unicode(raw) == "অা"


def test_indic_script_ratio_assamese():
    text = "অসমীয়া ভাষা"
    assert indic_script_ratio(text) > 0.8


def test_latin_script_ratio_english():
    text = "This is an English document about policy."
    assert latin_script_ratio(text) > 0.8


def test_detect_language_assamese():
    assert detect_language("অসমীয়া প্ৰশ্ন") == "as"


def test_detect_language_english():
    assert detect_language("What is the capital of Assam?") == "en"


def test_detect_language_mixed():
    text = "অসমীয়া text mixed with English words here"
    assert detect_language(text) == "mixed"


def test_detect_query_language():
    assert detect_query_language("What is GST?") == "en"
    assert detect_query_language("গুৱাহাটী ক'ত?") == "as"


def test_quality_score_english():
    text = "This is a clean English paragraph for embedding."
    assert quality_score(text) >= 0.45


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
    assert is_embeddable("This is enough English text to embed.", min_quality=0.45)
    assert not is_embeddable("x", min_quality=0.45)
