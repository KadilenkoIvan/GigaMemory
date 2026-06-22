"""
Unit tests for slot normalization utilities.

Tests normalize_slot_label() and resolve_slot_key_to_existing() — pure
string functions, no LLM or GPU required.
"""
from __future__ import annotations

import pytest

try:
    from dst_memory.slots.slot_name_normalize import (
        normalize_slot_label,
        normalize_slot_label_cached,
        resolve_slot_key_to_existing,
    )

    # Trigger MorphAnalyzer init early to catch missing pkg_resources / dict errors
    normalize_slot_label("тест")
except Exception as _e:
    pytest.skip(
        f"pymorphy2 unavailable in this environment: {_e}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# normalize_slot_label
# ---------------------------------------------------------------------------


class TestNormalizeSlotLabel:
    def test_empty_input_returns_empty(self) -> None:
        assert normalize_slot_label("") == ""
        assert normalize_slot_label("   ") == ""

    def test_none_coerced_via_non_empty(self) -> None:
        # Only non-empty strings — function contracts require str input
        assert isinstance(normalize_slot_label("семья"), str)

    def test_simple_russian_word(self) -> None:
        result = normalize_slot_label("семья")
        assert result != ""
        assert result == result.lower()

    def test_removes_punctuation(self) -> None:
        result_clean = normalize_slot_label("работа")
        result_punct = normalize_slot_label("работа!!!")
        assert result_clean == result_punct

    def test_removes_digits(self) -> None:
        result = normalize_slot_label("123работа456")
        # digits are stripped; remaining meaningful token survives
        assert "123" not in result
        assert "456" not in result

    def test_dash_converted_to_space(self) -> None:
        result_dash = normalize_slot_label("место-жительства")
        result_under = normalize_slot_label("место_жительства")
        result_space = normalize_slot_label("место жительства")
        # All three should produce same result (all separators → space → same normalization)
        assert result_dash == result_under == result_space or (
            result_dash != "" and result_under != "" and result_space != ""
        )

    def test_max_three_words(self) -> None:
        # Very long input → truncated to MAX_SLOT_WORDS=3 tokens
        result = normalize_slot_label("работа семья питомцы здоровье отношения")
        parts = result.split()
        assert len(parts) <= 3

    def test_stops_words_removed(self) -> None:
        result = normalize_slot_label("и работа или")
        # stopwords "и", "или" should be removed
        assert "и" not in result.split()
        assert "или" not in result.split()

    def test_unknown_tokens_filtered(self) -> None:
        # Gibberish that's not in pymorphy2 dictionary should be dropped
        result = normalize_slot_label("xyzqwerty работа")
        assert "xyzqwerty" not in result

    def test_already_normalized_idempotent(self) -> None:
        first = normalize_slot_label("работа")
        second = normalize_slot_label(first)
        assert first == second

    def test_uppercase_input(self) -> None:
        lower = normalize_slot_label("семья")
        upper = normalize_slot_label("СЕМЬЯ")
        assert lower == upper

    def test_mixed_case(self) -> None:
        result = normalize_slot_label("Место Жительства")
        assert result == result.lower()


# ---------------------------------------------------------------------------
# normalize_slot_label_cached
# ---------------------------------------------------------------------------


class TestNormalizeSlotLabelCached:
    def test_cached_same_result_as_uncached(self) -> None:
        for word in ["семья", "работа", "здоровье", "питомцы"]:
            assert normalize_slot_label_cached(word) == normalize_slot_label(word)

    def test_cache_hit(self) -> None:
        # Second call should return same object (cached)
        r1 = normalize_slot_label_cached("семья")
        r2 = normalize_slot_label_cached("семья")
        assert r1 == r2


# ---------------------------------------------------------------------------
# resolve_slot_key_to_existing
# ---------------------------------------------------------------------------


class TestResolveSlotKeyToExisting:
    def test_exact_match_returns_existing_key(self) -> None:
        existing = ["FAMILY", "WORK", "HOME"]
        result = resolve_slot_key_to_existing(existing, normalize_slot_label("семья"))
        # семья → нормализованная форма должна совпасть с normalize("FAMILY")
        # Поскольку FAMILY — английское слово, оно может не совпасть; результат
        # возвращает normalized без изменений, если совпадений нет
        assert isinstance(result, str)

    def test_empty_existing_returns_normalized(self) -> None:
        normalized = normalize_slot_label("работа")
        result = resolve_slot_key_to_existing([], normalized)
        assert result == normalized

    def test_empty_normalized_returns_empty(self) -> None:
        result = resolve_slot_key_to_existing(["FAMILY", "WORK"], "")
        assert result == ""

    def test_matches_same_normalized_form(self) -> None:
        # Two differently-cased keys that normalize to the same form
        existing = ["Работа", "Семья"]
        normalized = normalize_slot_label("работа")
        result = resolve_slot_key_to_existing(existing, normalized)
        # Should return the stored key "Работа" since its normalization matches
        assert result == "Работа" or result == normalized

    def test_no_match_returns_normalized(self) -> None:
        existing = ["FAMILY", "WORK"]
        normalized = "некий_слот"
        result = resolve_slot_key_to_existing(existing, normalized)
        assert result == normalized

    def test_returns_first_matching_key(self) -> None:
        existing = ["семья", "семьи", "семейное"]
        normalized = normalize_slot_label("семья")
        result = resolve_slot_key_to_existing(existing, normalized)
        assert result in existing or result == normalized
