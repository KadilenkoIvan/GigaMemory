"""
Tests for entity and relation merge logic.
"""

import pytest
from ragu.graph.types import Entity, Relation
from ragu.graph.index import Index, StorageArguments
from ragu.models.embedder import Embedder
from unittest.mock import AsyncMock


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    embedder = AsyncMock(spec=Embedder)
    embedder.dim = 128
    embedder.embed_text = AsyncMock(return_value=[0.1] * 128)
    return embedder


@pytest.fixture
def index(tmp_path, mock_embedder):
    """Create an Index instance."""
    storage_args = StorageArguments()
    index = Index(embedder=mock_embedder, arguments=storage_args)
    return index


def test_merge_entities_no_duplicates(index):
    """Test merging when there are no duplicates."""
    entity1 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Description 1",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    entity2 = Entity(
        id="ent-2",
        entity_name="Bob",
        entity_type="Person",
        description="Description 2",
        source_chunk_id=["chunk-2"],
        documents_id=["doc-2"],
        clusters=[],
    )

    groups = {
        ("Alice", "Person"): [entity1],
        ("Bob", "Person"): [entity2],
    }

    merged = index._merge_entities(groups)

    assert len(merged) == 2
    assert merged[0].id == "ent-1"
    assert merged[1].id == "ent-2"


def test_merge_entities_with_duplicates(index):
    """Test merging entities with duplicates."""
    entity1 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Software engineer",
        source_chunk_id=["chunk-1", "chunk-2"],
        documents_id=["doc-1"],
        clusters=[{"level": 0, "cluster_id": 1}],
    )
    entity2 = Entity(
        id="ent-2",
        entity_name="Alice",
        entity_type="Person",
        description="Works at Acme Corp",
        source_chunk_id=["chunk-3"],
        documents_id=["doc-2"],
        clusters=[],
    )

    groups = {
        ("Alice", "Person"): [entity1, entity2],
    }

    merged = index._merge_entities(groups)

    assert len(merged) == 1
    merged_entity = merged[0]

    # Should use primary ID (from entity with most chunks)
    assert merged_entity.id == "ent-1"

    # Should merge descriptions
    assert "Software engineer" in merged_entity.description
    assert "Works at Acme Corp" in merged_entity.description

    # Should union source chunks
    assert set(merged_entity.source_chunk_id) == {"chunk-1", "chunk-2", "chunk-3"}

    # Should union documents
    assert set(merged_entity.documents_id) == {"doc-1", "doc-2"}


def test_merge_entities_sorts_by_richness(index):
    """Test that merge uses entity with most chunks as primary."""
    entity1 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Description 1",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    entity2 = Entity(
        id="ent-2",
        entity_name="Alice",
        entity_type="Person",
        description="Description 2",
        source_chunk_id=["chunk-2", "chunk-3", "chunk-4"],
        documents_id=["doc-2"],
        clusters=[],
    )

    groups = {
        ("Alice", "Person"): [entity1, entity2],
    }

    merged = index._merge_entities(groups)

    # Should use ent-2 as primary (has more chunks)
    assert merged[0].id == "ent-2"


def test_merge_relations_no_duplicates(index):
    """Test merging relations with no duplicates."""
    rel1 = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Alice knows Bob",
        relation_strength=1.0,
        source_chunk_id=["chunk-1"],
    )
    rel2 = Relation(
        id="rel-2",
        subject_id="ent-2",
        object_id="ent-3",
        subject_name="Bob",
        object_name="Charlie",
        relation_type="KNOWS",
        description="Bob knows Charlie",
        relation_strength=1.0,
        source_chunk_id=["chunk-2"],
    )

    groups = {
        ("ent-1", "ent-2", "KNOWS"): [rel1],
        ("ent-2", "ent-3", "KNOWS"): [rel2],
    }

    merged = index._merge_relations(groups)

    assert len(merged) == 2


def test_merge_relations_with_duplicates(index):
    """Test merging duplicate relations."""
    rel1 = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="WORKS_WITH",
        description="They work together",
        relation_strength=1.0,
        source_chunk_id=["chunk-1", "chunk-2"],
    )
    rel2 = Relation(
        id="rel-2",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="WORKS_WITH",
        description="Colleagues on the same project",
        relation_strength=0.8,
        source_chunk_id=["chunk-3"],
    )

    groups = {
        ("ent-1", "ent-2", "WORKS_WITH"): [rel1, rel2],
    }

    merged = index._merge_relations(groups)

    assert len(merged) == 1
    merged_rel = merged[0]

    # Should use primary ID
    assert merged_rel.id == "rel-1"

    # Should merge descriptions
    assert "They work together" in merged_rel.description
    assert "Colleagues on the same project" in merged_rel.description

    # Should average strength
    assert merged_rel.relation_strength == pytest.approx(0.9)

    # Should union source chunks
    assert set(merged_rel.source_chunk_id) == {"chunk-1", "chunk-2", "chunk-3"}


def test_merge_entities_deduplicates_descriptions(index):
    """Test that duplicate descriptions are not repeated."""
    entity1 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Software engineer",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    entity2 = Entity(
        id="ent-2",
        entity_name="Alice",
        entity_type="Person",
        description="Software engineer",  # Same description
        source_chunk_id=["chunk-2"],
        documents_id=["doc-1"],
        clusters=[],
    )

    groups = {
        ("Alice", "Person"): [entity1, entity2],
    }

    merged = index._merge_entities(groups)

    # Should not duplicate description
    assert merged[0].description == "Software engineer"


def test_merge_relations_deduplicates_descriptions(index):
    """Test that duplicate relation descriptions are not repeated."""
    rel1 = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Friends",
        relation_strength=1.0,
        source_chunk_id=["chunk-1"],
    )
    rel2 = Relation(
        id="rel-2",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Friends",  # Same description
        relation_strength=1.0,
        source_chunk_id=["chunk-2"],
    )

    groups = {
        ("ent-1", "ent-2", "KNOWS"): [rel1, rel2],
    }

    merged = index._merge_relations(groups)

    # Should not duplicate description
    assert merged[0].description == "Friends"


def test_merge_is_deterministic(index):
    """Test that merge produces consistent results."""
    entity1 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Desc A",
        source_chunk_id=["chunk-1", "chunk-2"],
        documents_id=["doc-1"],
        clusters=[],
    )
    entity2 = Entity(
        id="ent-2",
        entity_name="Alice",
        entity_type="Person",
        description="Desc B",
        source_chunk_id=["chunk-3"],
        documents_id=["doc-2"],
        clusters=[],
    )

    groups = {
        ("Alice", "Person"): [entity1, entity2],
    }

    # Merge multiple times
    merged1 = index._merge_entities(groups)
    merged2 = index._merge_entities(groups)

    # Results should be identical
    assert merged1[0].id == merged2[0].id
    assert merged1[0].description == merged2[0].description
    assert merged1[0].source_chunk_id == merged2[0].source_chunk_id


def test_merge_entities_deduplicates_description_fragments(index):
    entity1 = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Software engineer. Works at Acme.",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    entity2 = Entity(
        id="ent-2",
        entity_name="Alice",
        entity_type="Person",
        description="Software engineer.",
        source_chunk_id=["chunk-2"],
        documents_id=["doc-2"],
        clusters=[],
    )

    merged = index._merge_entities({("Alice", "Person"): [entity1, entity2]})

    assert merged[0].description.count("Software engineer.") == 1


def test_merge_relations_deduplicates_description_fragments(index):
    rel1 = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Friends. Work together.",
        relation_strength=1.0,
        source_chunk_id=["chunk-1"],
    )
    rel2 = Relation(
        id="rel-2",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Friends.",
        relation_strength=1.0,
        source_chunk_id=["chunk-2"],
    )

    merged = index._merge_relations({"rel-key": [rel1, rel2]})

    assert merged[0].description.count("Friends.") == 1
