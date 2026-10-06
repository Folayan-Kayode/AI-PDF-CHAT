"""Tests for the text helpers."""

import pytest

from app.utils.helpers import clean_text, estimate_tokens

# --------------------------------------------------------------------------
# clean_text
# --------------------------------------------------------------------------


def test_clean_text_collapses_spaces_within_a_line():
    assert clean_text("a    b\t\tc") == "a b c"


def test_clean_text_preserves_line_structure():
    # Tables, forms and lists are the documents most likely to be ingested by
    # accident, and flattening them destroys the only structure they have.
    text = "Item      Qty\nCable     2\nAdapter   1"

    assert clean_text(text) == "Item Qty\nCable 2\nAdapter 1"


def test_clean_text_drops_blank_lines():
    assert clean_text("a\n\n\n   \nb") == "a\nb"


def test_clean_text_normalises_windows_and_old_mac_line_endings():
    assert clean_text("a\r\nb\rc") == "a\nb\nc"


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""
    assert clean_text("   \n  ") == ""


# --------------------------------------------------------------------------
# estimate_tokens
# --------------------------------------------------------------------------


def test_estimate_tokens_of_latin_text_is_roughly_a_quarter():
    assert estimate_tokens("a" * 400) == 100


def test_estimate_tokens_of_empty_text_is_zero():
    assert estimate_tokens("") == 0


@pytest.mark.parametrize("text", ["日本語のテキスト", "한국어 텍스트", "中文文本"])
def test_estimate_tokens_counts_cjk_characters_as_tokens(text):
    # A character budget alone would under-count these by roughly 4x.
    assert estimate_tokens(text) >= len(text) - 4


def test_cjk_costs_far_more_than_the_same_latin_length():
    latin = estimate_tokens("a" * 100)
    cjk = estimate_tokens("中" * 100)

    assert cjk > latin * 3
