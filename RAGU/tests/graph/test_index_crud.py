"""
Tests for Index batch CRUD operations and invariant validation.
"""

import pytest
from ragu.graph.types import Entity, Relation
from ragu.chunker.types import Chunk
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
    """Create an Index instance with temporary storage."""
    from ragu.common.global_parameters import Settings
    Settings.storage_folder = str(tmp_path / "storage")
    storage_args = StorageArguments()
    return Index(embedder=mock_embedder, arguments=storage_args)


@pytest.fixture
def sample_entities():
    """Sample entities for testing."""
    return [
        Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="A software engineer",
            source_chunk_id=["chunk-1"],
            documents_id=["doc-1"],
            clusters=[],
        ),
        Entity(
            id="ent-2",
            entity_name="Bob",
            entity_type="Person",
            description="A data scientist",
            source_chunk_id=["chunk-2"],
            documents_id=["doc-1"],
            clusters=[],
        ),
    ]


@pytest.fixture
def sample_relations(sample_entities):
    """Sample relations for testing."""
    return [
        Relation(
            id="rel-1",
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="Alice",
            object_name="Bob",
            relation_type="KNOWS",
            description="Alice knows Bob",
            relation_strength=1.0,
            source_chunk_id=["chunk-1"],
        ),
    ]


@pytest.mark.asyncio
async def test_insert_entities(index, sample_entities):
    """
    Test inserting entities.
    """
    await index.insert_entities(sample_entities)

    # Verify entities exist
    retrieved = await index.get_entities(["ent-1", "ent-2"])
    assert len(retrieved) == 2
    assert all(e is not None for e in retrieved)


@pytest.mark.asyncio
async def test_upsert_entities_with_merge(index):
    """
    Test upserting duplicate entities merges them.
    """
    entity1 = Entity(
        entity_name="Alice",
        entity_type="Person",
        description="First description",
        source_chunk_id=["chunk-1"],
        documents_id=["doc-1"],
        clusters=[],
    )
    entity2 = Entity(
        entity_name="Alice",
        entity_type="Person",
        description="Second description",
        source_chunk_id=["chunk-2"],
        documents_id=["doc-2"],
        clusters=[],
    )

    # First insert
    await index.insert_entities([entity1])

    # Second insert with duplicate
    await index.insert_entities([entity2])

    # Should have merged - only one Alice entity
    retrieved = await index.get_entities([entity1.id])
    non_null = [e for e in retrieved if e is not None]
    assert len(non_null) == 1

    # Merged entity should have both descriptions
    merged = non_null[0]
    assert "First description" in merged.description
    assert "Second description" in merged.description


@pytest.mark.asyncio
async def test_insert_relations(index, sample_entities, sample_relations):
    """
    Test inserting relations.
    """
    await index.insert_entities(sample_entities)
    await index.insert_relations(sample_relations)

    relation = sample_relations[0]
    retrieved = await index.get_relations([(relation.subject_id, relation.object_id, relation.id)])
    assert len(retrieved) == 1
    assert retrieved[0] is not None


@pytest.mark.asyncio
async def test_upsert_relations_validates_entities(index, sample_relations):
    """
    Test that upserting relations validates entity existence.
    """
    with pytest.raises(ValueError, match="non-existent entities"):
        await index.insert_relations(sample_relations)


@pytest.mark.asyncio
async def test_delete_entities(index, sample_entities):
    """
    Test deleting entities.
    """
    await index.insert_entities(sample_entities)
    await index.delete_entities(["ent-1"])

    retrieved = await index.get_entities(["ent-1", "ent-2"])
    assert retrieved[0] is None
    assert retrieved[1] is not None


@pytest.mark.asyncio
async def test_delete_entities_cascade(index, sample_entities, sample_relations):
    """
    Test that deleting entities cascades to relations.
    """
    await index.insert_entities(sample_entities)
    await index.insert_relations(sample_relations)

    await index.delete_entities(["ent-1"])

    relation = sample_relations[0]
    retrieved_relations = await index.get_relations([(relation.subject_id, relation.object_id, relation.id)])
    assert retrieved_relations[0] is None


@pytest.mark.asyncio
async def test_delete_relations(index, sample_entities, sample_relations):
    """
    Test deleting relations.
    """
    await index.insert_entities(sample_entities)
    await index.insert_relations(sample_relations)

    relation = sample_relations[0]
    await index.delete_relations([(relation.subject_id, relation.object_id, relation.id)])

    retrieved = await index.get_relations([(relation.subject_id, relation.object_id, relation.id)])
    assert retrieved[0] is None

    entities = await index.get_entities(["ent-1", "ent-2"])
    assert all(e is not None for e in entities)


@pytest.mark.asyncio
async def test_upsert_relations_keeps_non_duplicate_when_duplicates_exist(index, sample_entities):
    await index.insert_entities(sample_entities)

    rel_existing = Relation(
        id="rel-dup",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="old",
        relation_strength=1.0,
        source_chunk_id=["chunk-1"],
    )
    await index.insert_relations([rel_existing])

    rel_duplicate_update = Relation(
        id="rel-dup",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="new",
        relation_strength=1.0,
        source_chunk_id=["chunk-2"],
    )
    rel_unique_new = Relation(
        id="rel-new",
        subject_id="ent-2",
        object_id="ent-1",
        subject_name="Bob",
        object_name="Alice",
        relation_type="KNOWS",
        description="fresh",
        relation_strength=1.0,
        source_chunk_id=["chunk-2"],
    )

    await index.insert_relations([rel_duplicate_update, rel_unique_new])

    got = await index.get_relations(
        [
            ("ent-1", "ent-2", "rel-dup"),
            ("ent-2", "ent-1", "rel-new"),
        ]
    )
    assert got[0] is not None
    assert got[1] is not None


@pytest.mark.asyncio
async def test_update_entities_replaces_existing_payload(index):
    original = Entity(
        id="ent-update",
        entity_name="Alice",
        entity_type="Person",
        description="old description",
        source_chunk_id=["chunk-old"],
        documents_id=["doc-old"],
        clusters=[{"level": 1, "cluster_id": "1"}],
    )
    updated = Entity(
        id="ent-update",
        entity_name="Alice Updated",
        entity_type="Person",
        description="new description",
        source_chunk_id=["chunk-new"],
        documents_id=["doc-new"],
        clusters=[{"level": 2, "cluster_id": "2"}],
    )

    await index.insert_entities([original])
    await index.update_entities([updated])

    got = await index.get_entities([original.id])
    assert got[0] is not None
    assert got[0].description == "new description"
    assert got[0].entity_name == "Alice Updated"
    assert got[0].source_chunk_id == ["chunk-new"]
    assert got[0].documents_id == ["doc-new"]
    assert got[0].clusters == [{"level": 2, "cluster_id": "2"}]


@pytest.mark.asyncio
async def test_update_entities_fails_for_missing_id(index):
    missing = Entity(
        id="ent-missing",
        entity_name="Ghost",
        entity_type="Person",
        description="does not exist",
        source_chunk_id=["chunk-x"],
        documents_id=[],
        clusters=[],
    )

    with pytest.raises(ValueError, match="non-existent entities"):
        await index.update_entities([missing])


@pytest.mark.asyncio
async def test_update_relations_replaces_existing_payload(index):
    entities = [
        Entity(id="ent-a", entity_name="A", entity_type="Node", description="A", source_chunk_id=["chunk-a"]),
        Entity(id="ent-b", entity_name="B", entity_type="Node", description="B", source_chunk_id=["chunk-b"]),
        Entity(id="ent-c", entity_name="C", entity_type="Node", description="C", source_chunk_id=["chunk-c"]),
    ]
    await index.insert_entities(entities)

    original = Relation(
        id="rel-update",
        subject_id="ent-a",
        object_id="ent-b",
        subject_name="A",
        object_name="B",
        relation_type="LINKS",
        description="old relation",
        relation_strength=1.0,
        source_chunk_id=["chunk-old"],
    )
    updated = Relation(
        id="rel-update",
        subject_id="ent-b",
        object_id="ent-c",
        subject_name="B",
        object_name="C",
        relation_type="LINKS",
        description="new relation",
        relation_strength=7.0,
        source_chunk_id=["chunk-new"],
    )

    await index.insert_relations([original])
    await index.update_relations([updated])

    old_edge = await index.get_relations([("ent-a", "ent-b", "rel-update")])
    new_edge = await index.get_relations([("ent-b", "ent-c", "rel-update")])

    assert old_edge[0] is None
    assert new_edge[0] is not None
    assert new_edge[0].description == "new relation"
    assert new_edge[0].relation_strength == 7.0
    assert new_edge[0].source_chunk_id == ["chunk-new"]


@pytest.mark.asyncio
async def test_update_relations_fails_for_missing_id(index, sample_entities):
    await index.insert_entities(sample_entities)

    missing_relation = Relation(
        id="rel-missing",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="missing relation",
        relation_strength=1.0,
        source_chunk_id=["chunk-x"],
    )

    with pytest.raises(ValueError, match="non-existent relations"):
        await index.update_relations([missing_relation])


@pytest.mark.asyncio
async def test_upsert_chunks(index):
    """
    Test upserting chunks.
    """
    chunk1 = Chunk(content="Some text", chunk_order_idx=0, doc_id="doc-1")
    chunk2 = Chunk(content="More text", chunk_order_idx=1, doc_id="doc-1")
    chunks = [chunk1, chunk2]

    await index.upsert_chunks(chunks)

    # Verify chunks exist
    retrieved = await index.get_chunks([chunk1.id, chunk2.id])
    assert len(retrieved) == 2
    assert all(c is not None for c in retrieved)


@pytest.mark.asyncio
async def test_delete_chunks_cascade(index):
    """
    Test deleting chunks cascades to related entities and relations.
    """
    chunk1 = Chunk(content="Chunk one text", chunk_order_idx=0, doc_id="doc-1")
    chunk2 = Chunk(content="Chunk two text", chunk_order_idx=1, doc_id="doc-1")
    chunk3 = Chunk(content="Chunk three text", chunk_order_idx=2, doc_id="doc-1")

    entities = [
        Entity(
            id="ent-1",
            entity_name="Alice",
            entity_type="Person",
            description="Alice",
            source_chunk_id=[chunk1.id],
            documents_id=["doc-1"],
            clusters=[],
        ),
        Entity(
            id="ent-2",
            entity_name="Bob",
            entity_type="Person",
            description="Bob",
            source_chunk_id=[chunk2.id],
            documents_id=["doc-1"],
            clusters=[],
        ),
        Entity(
            id="ent-3",
            entity_name="Charlie",
            entity_type="Person",
            description="Charlie",
            source_chunk_id=[chunk3.id],
            documents_id=["doc-1"],
            clusters=[],
        ),
    ]

    relations = [
        Relation(
            id="rel-1",
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="Alice",
            object_name="Bob",
            relation_type="KNOWS",
            description="Alice knows Bob",
            relation_strength=1.0,
            source_chunk_id=[chunk1.id],
        ),
        Relation(
            id="rel-2",
            subject_id="ent-2",
            object_id="ent-3",
            subject_name="Bob",
            object_name="Charlie",
            relation_type="KNOWS",
            description="Bob knows Charlie",
            relation_strength=1.0,
            source_chunk_id=[chunk2.id],
        ),
    ]

    await index.insert_entities(entities)
    await index.insert_relations(relations)
    await index.upsert_chunks([chunk1, chunk2, chunk3])

    await index.delete_chunks([chunk1.id])

    chunks_after_delete = await index.get_chunks([chunk1.id, chunk2.id, chunk3.id])
    assert chunks_after_delete[0] is None
    assert chunks_after_delete[1] is not None
    assert chunks_after_delete[2] is not None

    entities_after_delete = await index.get_entities(["ent-1", "ent-2", "ent-3"])
    assert entities_after_delete[0] is None
    assert entities_after_delete[1] is not None
    assert entities_after_delete[2] is not None

    relations_after_delete = await index.get_relations(
        [
            ("ent-1", "ent-2", "rel-1"),
            ("ent-2", "ent-3", "rel-2"),
        ]
    )
    assert relations_after_delete[0] is None
    assert relations_after_delete[1] is not None
