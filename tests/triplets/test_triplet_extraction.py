"""
Unit tests for TripletExtractionClient.

All tests use use_stub=True (no GPU / LLM needed) or test the pure _parse()
logic directly — no model inference happens.
"""
from __future__ import annotations

import pytest

from dst_memory.triplets.triplet_client import (
    DeletionSignal,
    ExtractedTriplet,
    TripletExtractionClient,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_client() -> TripletExtractionClient:
    """TripletExtractionClient in stub mode — returns empty lists."""
    return TripletExtractionClient(use_stub=True)


# ---------------------------------------------------------------------------
# Stub mode behaviour
# ---------------------------------------------------------------------------


def test_stub_extract_returns_empty(stub_client: TripletExtractionClient) -> None:
    triplets = stub_client.extract("Меня зовут Иван, мне 30 лет")
    assert triplets == []


def test_stub_extract_for_slot_returns_empty(stub_client: TripletExtractionClient) -> None:
    triplets = stub_client.extract_for_slot("Я живу в Москве", "LOCATION")
    assert triplets == []


def test_stub_extract_with_context_returns_empty(stub_client: TripletExtractionClient) -> None:
    triplets, deletions = stub_client.extract_with_context(
        "Я переехал в Питер",
        existing_triplets=["пользователь | место жительства | москва"],
    )
    assert triplets == []
    assert deletions == []


# ---------------------------------------------------------------------------
# _parse: valid JSON → ExtractedTriplet list
# ---------------------------------------------------------------------------


def test_parse_valid_triplets(stub_client: TripletExtractionClient) -> None:
    raw = '{"triplets":[{"slot":"IDENTITY","subject":"пользователь","relation":"имя","object":"иван","ttl":"inf"}]}'
    result = stub_client._parse(raw, forced_slot="IDENTITY")
    assert result is not None
    triplets, deletions = result
    assert len(triplets) == 1
    assert triplets[0].subject == "пользователь"
    assert triplets[0].relation == "имя"
    assert triplets[0].object == "иван"
    assert triplets[0].ttl == "inf"
    assert deletions == []


def test_parse_multiple_triplets(stub_client: TripletExtractionClient) -> None:
    raw = """{"triplets":[
        {"slot":"FAMILY","subject":"пользователь","relation":"есть сын","object":"сын пользователя","ttl":"inf"},
        {"slot":"FAMILY","subject":"сын пользователя","relation":"имя","object":"алёша","ttl":"1y"}
    ]}"""
    result = stub_client._parse(raw, forced_slot="FAMILY")
    assert result is not None
    triplets, _ = result
    assert len(triplets) == 2
    assert triplets[1].object == "алёша"
    assert triplets[1].ttl == "1y"


def test_parse_with_markdown_fences(stub_client: TripletExtractionClient) -> None:
    raw = '```json\n{"triplets":[{"slot":"WORK","subject":"пользователь","relation":"работает в","object":"яндекс","ttl":"1y"}]}\n```'
    result = stub_client._parse(raw, forced_slot="WORK")
    assert result is not None
    triplets, _ = result
    assert len(triplets) == 1
    assert triplets[0].object == "яндекс"


def test_parse_empty_triplets_list(stub_client: TripletExtractionClient) -> None:
    raw = '{"triplets":[]}'
    result = stub_client._parse(raw)
    assert result is not None
    triplets, deletions = result
    assert triplets == []
    assert deletions == []


def test_parse_invalid_json_returns_none(stub_client: TripletExtractionClient) -> None:
    assert stub_client._parse("not json at all") is None
    assert stub_client._parse("") is None
    assert stub_client._parse("{}") is None


def test_parse_invalid_ttl_falls_back_to_inf(stub_client: TripletExtractionClient) -> None:
    raw = '{"triplets":[{"slot":"FOOD","subject":"пользователь","relation":"ест","object":"пиццу","ttl":"bad_value"}]}'
    result = stub_client._parse(raw, forced_slot="FOOD")
    assert result is not None
    triplets, _ = result
    assert triplets[0].ttl == "inf"


def test_parse_all_valid_ttl_values(stub_client: TripletExtractionClient) -> None:
    valid_ttls = ["1d", "3d", "10d", "2w", "3w", "1m", "3m", "6m", "1y", "inf"]
    for ttl in valid_ttls:
        raw = f'{{"triplets":[{{"slot":"IDENTITY","subject":"u","relation":"r","object":"o","ttl":"{ttl}"}}]}}'
        result = stub_client._parse(raw, forced_slot="IDENTITY")
        assert result is not None
        assert result[0][0].ttl == ttl, f"TTL {ttl!r} was not preserved"


def test_parse_skips_incomplete_triplets(stub_client: TripletExtractionClient) -> None:
    raw = '{"triplets":[{"slot":"IDENTITY","subject":"","relation":"имя","object":"иван","ttl":"inf"}]}'
    result = stub_client._parse(raw, forced_slot="IDENTITY")
    assert result is not None
    triplets, _ = result
    # subject is empty → should be skipped
    assert triplets == []


# ---------------------------------------------------------------------------
# _parse: deletion signals (with_deletions=True)
# ---------------------------------------------------------------------------


def test_parse_with_deletion_signals(stub_client: TripletExtractionClient) -> None:
    raw = """{
        "triplets":[{"slot":"HOME","subject":"пользователь","relation":"живёт в","object":"питер","ttl":"1y"}],
        "delete":[{"subject":"пользователь","relation":"живёт в","object":"москва"}]
    }"""
    result = stub_client._parse(raw, forced_slot="HOME", with_deletions=True)
    assert result is not None
    triplets, deletions = result
    assert len(triplets) == 1
    assert triplets[0].object == "питер"
    assert len(deletions) == 1
    assert deletions[0].subject == "пользователь"
    assert deletions[0].relation == "живёт в"
    assert deletions[0].object == "москва"


def test_parse_delete_field_ignored_when_flag_false(stub_client: TripletExtractionClient) -> None:
    raw = """{
        "triplets":[{"slot":"HOME","subject":"пользователь","relation":"живёт в","object":"питер","ttl":"1y"}],
        "delete":[{"subject":"пользователь","relation":"живёт в","object":"москва"}]
    }"""
    result = stub_client._parse(raw, forced_slot="HOME", with_deletions=False)
    assert result is not None
    _, deletions = result
    assert deletions == []


# ---------------------------------------------------------------------------
# _normalize_field
# ---------------------------------------------------------------------------


def test_normalize_field_strips_and_lowercases() -> None:
    assert TripletExtractionClient._normalize_field("  Иван  ") == "иван"
    assert TripletExtractionClient._normalize_field("МОСКВА") == "москва"
    assert TripletExtractionClient._normalize_field("место  жительства") == "место жительства"


def test_normalize_field_collapses_whitespace() -> None:
    result = TripletExtractionClient._normalize_field("a   b   c")
    assert result == "a b c"


# ---------------------------------------------------------------------------
# ExtractedTriplet and DeletionSignal helpers
# ---------------------------------------------------------------------------


def test_extracted_triplet_as_line() -> None:
    t = ExtractedTriplet(slot="IDENTITY", subject="u", relation="r", object="o")
    assert t.as_line() == "u | r | o"


def test_deletion_signal_as_line() -> None:
    d = DeletionSignal(subject="пользователь", relation="живёт в", object="москва")
    assert d.as_line() == "пользователь | живёт в | москва"
