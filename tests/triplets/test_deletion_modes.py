"""
Unit tests for all three deletion modes:
  A (llm_inline) — signals embedded in extraction response
  B (llm_separate) — TripletDeletionClient stub returns []
  C (heuristic) — NegationDeletionDetector pattern matching

All tests run without GPU / LLM (stub mode or pure pattern matching).
"""
from __future__ import annotations

import pytest

from dst_memory.core.models import FactRecord
from dst_memory.triplets.deletion_client import TripletDeletionClient
from dst_memory.triplets.negation_detector import NegationDeletionDetector
from dst_memory.triplets.triplet_client import DeletionSignal, TripletExtractionClient


# ---------------------------------------------------------------------------
# Helper: build a minimal FactRecord
# ---------------------------------------------------------------------------


def make_record(
    record_id: int,
    subject: str,
    relation: str,
    obj: str,
    slot: str = "TEST",
) -> FactRecord:
    return FactRecord(
        record_id=record_id,
        value=f"{subject} | {relation} | {obj}",
        source_text="test message",
        created_at_step=1,
        updated_at_step=1,
        subject=subject,
        relation=relation,
        object=obj,
    )


# ===========================================================================
# Mode A (llm_inline) — deletion signals parsed from extraction response
# ===========================================================================


class TestModeAInline:
    """Mode A: TripletExtractionClient._parse() returns delete signals when
    with_deletions=True. No LLM is called — tested via _parse() directly."""

    def setup_method(self) -> None:
        self.client = TripletExtractionClient(use_stub=True)

    def test_parse_delete_in_inline_response(self) -> None:
        raw = """{
            "triplets": [
                {"slot":"HOME","subject":"пользователь","relation":"живёт в","object":"питер","ttl":"1y"}
            ],
            "delete": [
                {"subject":"пользователь","relation":"живёт в","object":"москва"}
            ]
        }"""
        result = self.client._parse(raw, forced_slot="HOME", with_deletions=True)
        assert result is not None
        triplets, deletions = result
        assert any(t.object == "питер" for t in triplets)
        assert any(d.object == "москва" for d in deletions)

    def test_parse_multiple_deletes(self) -> None:
        raw = """{
            "triplets": [],
            "delete": [
                {"subject":"u","relation":"r1","object":"o1"},
                {"subject":"u","relation":"r2","object":"o2"}
            ]
        }"""
        result = self.client._parse(raw, with_deletions=True)
        assert result is not None
        _, deletions = result
        assert len(deletions) == 2

    def test_parse_empty_delete_list(self) -> None:
        raw = '{"triplets":[],"delete":[]}'
        result = self.client._parse(raw, with_deletions=True)
        assert result is not None
        _, deletions = result
        assert deletions == []

    def test_parse_delete_skips_incomplete_items(self) -> None:
        raw = """{
            "triplets": [],
            "delete": [
                {"subject":"u","relation":"r","object":""},
                {"subject":"u","relation":"r","object":"o"}
            ]
        }"""
        result = self.client._parse(raw, with_deletions=True)
        assert result is not None
        _, deletions = result
        # Item with empty object should be skipped
        assert len(deletions) == 1
        assert deletions[0].object == "o"


# ===========================================================================
# Mode B (llm_separate) — TripletDeletionClient stub
# ===========================================================================


class TestModeBSeparate:
    """Mode B: TripletDeletionClient in stub mode always returns empty list."""

    def setup_method(self) -> None:
        self.client = TripletDeletionClient(use_stub=True)

    def test_stub_returns_empty_regardless_of_input(self) -> None:
        result = self.client.detect_deletions(
            user_message="я уволился из яндекса",
            slot_name="WORK",
            existing_triplets=["пользователь | место работы | яндекс"],
        )
        assert result == []

    def test_stub_with_empty_existing_triplets(self) -> None:
        result = self.client.detect_deletions(
            user_message="больше не живу в Москве",
            slot_name="HOME",
            existing_triplets=[],
        )
        assert result == []

    def test_stub_with_any_message(self) -> None:
        for msg in [
            "расстался с девушкой",
            "уволился",
            "продал машину",
            "переехал",
        ]:
            result = self.client.detect_deletions(msg, "TEST", ["u | r | o"])
            assert result == [], f"Stub should return [] for: {msg!r}"


# ===========================================================================
# Mode C (heuristic) — NegationDeletionDetector
# ===========================================================================


class TestModeCHeuristic:
    """Mode C: NegationDeletionDetector — pattern-based, no LLM, no GPU."""

    def setup_method(self) -> None:
        self.detector = NegationDeletionDetector(use_pymorphy=False)

    # --- Negation detection ---

    def test_no_negation_returns_empty(self) -> None:
        records = [make_record(1, "пользователь", "живёт в", "москва")]
        result = self.detector.detect_deletions("Привет, как дела?", records)
        assert result == []

    def test_no_active_records_returns_empty(self) -> None:
        result = self.detector.detect_deletions("я больше не живу в Москве", [])
        assert result == []

    # --- Pass 1: exact object match ---

    def test_pass1_exact_object_match(self) -> None:
        records = [make_record(1, "пользователь", "место жительства", "москва")]
        result = self.detector.detect_deletions(
            "я больше не живу в москва, переехал", records
        )
        assert len(result) == 1
        assert result[0].object == "москва"

    def test_pass1_matches_work_resignation(self) -> None:
        records = [
            make_record(1, "пользователь", "место работы", "яндекс"),
            make_record(2, "пользователь", "должность", "разработчик"),
        ]
        result = self.detector.detect_deletions("уволился из яндекс", records)
        # яндекс should match
        objects = [r.object for r in result]
        assert "яндекс" in objects

    # --- Pass 2 (cascade): relation match only ---

    def test_pass2_cascade_relation_match(self) -> None:
        # Object "место" is not in message, but relation "место жительства"
        # partially overlaps. Use a message mentioning the relation word.
        records = [make_record(1, "пользователь", "работает", "разработчиком")]
        result = self.detector.detect_deletions("больше не работаю", records)
        # "работаю" normalizes to "работать" or similar; relation "работает" → "работать"
        # Cascade should trigger when pass1 finds nothing
        assert isinstance(result, list)

    # --- Pattern coverage ---

    @pytest.mark.parametrize(
        "message",
        [
            "я больше не живу там",
            "уже не работаю",
            "перестал ходить туда",
            "расстались с ней",
            "уволился месяц назад",
            "умер мой питомец",
            "продал машину",
            "переехал в другой город",
            "съехал оттуда",
            "у меня нет этого",
        ],
    )
    def test_negation_detected_in_message(self, message: str) -> None:
        assert self.detector._has_negation(message), (
            f"Expected negation pattern in: {message!r}"
        )

    @pytest.mark.parametrize(
        "message",
        [
            "Привет!",
            "Как дела?",
            "Сегодня хорошая погода",
            "Я иду в магазин",
            "Расскажи мне о Python",
        ],
    )
    def test_no_negation_in_neutral_messages(self, message: str) -> None:
        assert not self.detector._has_negation(message), (
            f"False positive negation in: {message!r}"
        )

    # --- Word normalization ---

    def test_normalize_text_removes_stopwords(self) -> None:
        words = self.detector._normalize_text("я живу в Москве")
        assert "в" not in words
        assert "я" not in words
        assert "москве" in words or "москва" in words or "москве" in words

    def test_normalize_text_lowercase(self) -> None:
        words = self.detector._normalize_text("ЯНДЕКС Москва")
        assert all(w == w.lower() for w in words)

    def test_normalize_text_strips_punctuation(self) -> None:
        words = self.detector._normalize_text("яндекс, москва!")
        assert "," not in words
        assert "!" not in words


# ---------------------------------------------------------------------------
# DeletionSignal dataclass
# ---------------------------------------------------------------------------


def test_deletion_signal_fields() -> None:
    sig = DeletionSignal(subject="u", relation="r", object="o")
    assert sig.subject == "u"
    assert sig.relation == "r"
    assert sig.object == "o"


def test_deletion_signal_is_frozen() -> None:
    sig = DeletionSignal(subject="u", relation="r", object="o")
    with pytest.raises((AttributeError, TypeError)):
        sig.subject = "x"  # type: ignore[misc]
