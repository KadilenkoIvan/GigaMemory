"""
Unit tests for TripletConflictClient — rule layer only.

All tests use use_stub=True so no LLM is called.
The rule layer runs unconditionally (before LLM).
"""
from __future__ import annotations

import pytest

from dst_memory.core.graph_backend import GraphEdge
from dst_memory.triplets.conflict_client import ConflictResolution, TripletConflictClient
from dst_memory.triplets.triplet_client import ExtractedTriplet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_edge(
    record_id: int,
    subject: str,
    relation: str,
    obj: str,
    slot: str = "TEST",
) -> GraphEdge:
    return GraphEdge(
        edge_id=record_id,  # use same value for simplicity in tests
        slot=slot,
        subject=subject,
        relation=relation,
        object=obj,
        record_id=record_id,
    )


def make_triplet(
    subject: str,
    relation: str,
    obj: str,
    slot: str = "TEST",
    ttl: str = "inf",
) -> ExtractedTriplet:
    return ExtractedTriplet(slot=slot, subject=subject, relation=relation, object=obj, ttl=ttl)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_client() -> TripletConflictClient:
    """Conflict client in stub mode — LLM layer disabled."""
    return TripletConflictClient(use_stub=True, rule_same_relation_updates=True)


@pytest.fixture
def stub_client_no_rule_updates() -> TripletConflictClient:
    """Conflict client with rule_same_relation_updates=False."""
    return TripletConflictClient(use_stub=True, rule_same_relation_updates=False)


# ---------------------------------------------------------------------------
# No existing edges — no conflicts possible
# ---------------------------------------------------------------------------


def test_no_existing_edges_no_conflict(stub_client: TripletConflictClient) -> None:
    new_triplets = [make_triplet("пользователь", "имя", "иван")]
    result = stub_client.resolve("IDENTITY", [], new_triplets)
    assert result.deactivate_ids == []
    assert result.skip_new_indices == set()


def test_empty_new_triplets_no_conflict(stub_client: TripletConflictClient) -> None:
    edges = [make_edge(1, "пользователь", "имя", "иван")]
    result = stub_client.resolve("IDENTITY", edges, [])
    assert result.deactivate_ids == []
    assert result.skip_new_indices == set()


# ---------------------------------------------------------------------------
# Rule 1: exact duplicate (same S + R + O) → skip new
# ---------------------------------------------------------------------------


def test_exact_duplicate_skips_new(stub_client: TripletConflictClient) -> None:
    edges = [make_edge(1, "пользователь", "имя", "иван")]
    new_triplets = [make_triplet("пользователь", "имя", "иван")]

    result = stub_client.resolve("IDENTITY", edges, new_triplets)
    assert 0 in result.skip_new_indices
    assert result.deactivate_ids == []


def test_exact_duplicate_only_skips_matching(stub_client: TripletConflictClient) -> None:
    edges = [make_edge(1, "пользователь", "имя", "иван")]
    new_triplets = [
        make_triplet("пользователь", "имя", "иван"),   # duplicate → skip
        make_triplet("пользователь", "возраст", "30"),  # new fact → keep
    ]

    result = stub_client.resolve("IDENTITY", edges, new_triplets)
    assert 0 in result.skip_new_indices
    assert 1 not in result.skip_new_indices


# ---------------------------------------------------------------------------
# Rule 2: same S + R, different O → deactivate old (rule_same_relation_updates=True)
# ---------------------------------------------------------------------------


def test_same_relation_different_object_deactivates_old(stub_client: TripletConflictClient) -> None:
    edges = [make_edge(10, "пользователь", "живёт в", "москва")]
    new_triplets = [make_triplet("пользователь", "живёт в", "питер")]

    result = stub_client.resolve("HOME", edges, new_triplets)
    assert 10 in result.deactivate_ids
    assert 0 not in result.skip_new_indices


def test_same_relation_deactivates_all_matching_edges(stub_client: TripletConflictClient) -> None:
    edges = [
        make_edge(10, "пользователь", "работает в", "яндекс"),
        make_edge(11, "пользователь", "работает в", "mail"),
    ]
    new_triplets = [make_triplet("пользователь", "работает в", "google")]

    result = stub_client.resolve("WORK", edges, new_triplets)
    assert 10 in result.deactivate_ids
    assert 11 in result.deactivate_ids


# ---------------------------------------------------------------------------
# Rule 2 disabled: rule_same_relation_updates=False
# ---------------------------------------------------------------------------


def test_same_relation_deferred_to_llm_when_rule_disabled(
    stub_client_no_rule_updates: TripletConflictClient,
) -> None:
    edges = [make_edge(10, "пользователь", "живёт в", "москва")]
    new_triplets = [make_triplet("пользователь", "живёт в", "питер")]

    result = stub_client_no_rule_updates.resolve("HOME", edges, new_triplets)
    # In stub mode LLM is disabled, but rule is also disabled → nothing deactivated
    assert 10 not in result.deactivate_ids
    assert 0 not in result.skip_new_indices


# ---------------------------------------------------------------------------
# allow_multi_relation_same_object: complementary facts
# ---------------------------------------------------------------------------


def test_complementary_facts_not_conflicted() -> None:
    client = TripletConflictClient(
        use_stub=True,
        allow_multi_relation_same_object=True,
    )
    edges = [make_edge(1, "пользователь", "есть партнёр", "партнёр пользователя")]
    new_triplets = [make_triplet("пользователь", "живёт вместе с", "партнёр пользователя")]

    result = client.resolve("ROMANCE", edges, new_triplets)
    # Same subject, same object, different relation → complementary → no conflict
    assert result.deactivate_ids == []
    assert 0 not in result.skip_new_indices


def test_complementary_facts_always_llm_when_flag_false() -> None:
    client = TripletConflictClient(
        use_stub=True,
        allow_multi_relation_same_object=False,
    )
    edges = [make_edge(1, "пользователь", "есть партнёр", "партнёр пользователя")]
    new_triplets = [make_triplet("пользователь", "живёт вместе с", "партнёр пользователя")]

    result = client.resolve("ROMANCE", edges, new_triplets)
    # LLM would be called, but stub → no changes
    assert result.deactivate_ids == []


# ---------------------------------------------------------------------------
# ConflictResolution dataclass
# ---------------------------------------------------------------------------


def test_conflict_resolution_defaults() -> None:
    cr = ConflictResolution(deactivate_ids=[], skip_new_indices=set())
    assert cr.deactivate_ids == []
    assert cr.skip_new_indices == set()


def test_conflict_resolution_fields() -> None:
    cr = ConflictResolution(deactivate_ids=[1, 2], skip_new_indices={0, 3})
    assert 1 in cr.deactivate_ids
    assert 0 in cr.skip_new_indices


# ---------------------------------------------------------------------------
# Multiple new triplets — partial conflict
# ---------------------------------------------------------------------------


def test_partial_conflict_among_multiple_new(stub_client: TripletConflictClient) -> None:
    edges = [make_edge(5, "пользователь", "имя", "иван")]
    new_triplets = [
        make_triplet("пользователь", "имя", "иван"),    # exact duplicate → skip
        make_triplet("пользователь", "возраст", "30"),   # no conflict → keep
        make_triplet("пользователь", "город", "москва"), # no conflict → keep
    ]

    result = stub_client.resolve("IDENTITY", edges, new_triplets)
    assert 0 in result.skip_new_indices
    assert 1 not in result.skip_new_indices
    assert 2 not in result.skip_new_indices
    assert result.deactivate_ids == []
