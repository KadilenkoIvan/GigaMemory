import pytest

from ragu.graph.builder_modules import RemoveIsolatedNodes
from ragu.graph.types import Entity, Relation


@pytest.mark.asyncio
async def test_remove_isolated_nodes_keeps_only_connected_entities():
    module = RemoveIsolatedNodes()

    alice = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Alice",
        source_chunk_id=["chunk-1"],
    )
    bob = Entity(
        id="ent-2",
        entity_name="Bob",
        entity_type="Person",
        description="Bob",
        source_chunk_id=["chunk-1"],
    )
    isolated = Entity(
        id="ent-3",
        entity_name="Carol",
        entity_type="Person",
        description="Carol",
        source_chunk_id=["chunk-2"],
    )

    relation = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-2",
        subject_name="Alice",
        object_name="Bob",
        relation_type="KNOWS",
        description="Alice knows Bob",
    )

    entities, relations = await module.run([alice, bob, isolated], [relation])

    assert [e.id for e in entities] == ["ent-1", "ent-2"]
    assert [r.id for r in relations] == ["rel-1"]


@pytest.mark.asyncio
async def test_remove_isolated_nodes_drops_relations_with_missing_endpoints():
    module = RemoveIsolatedNodes()

    alice = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Alice",
        source_chunk_id=["chunk-1"],
    )

    dangling = Relation(
        id="rel-1",
        subject_id="ent-1",
        object_id="ent-999",
        subject_name="Alice",
        object_name="Unknown",
        relation_type="KNOWS",
        description="Dangling relation",
    )

    entities, relations = await module.run([alice], [dangling])

    assert entities == []
    assert relations == []


@pytest.mark.asyncio
async def test_remove_isolated_nodes_with_no_relations_returns_no_entities():
    module = RemoveIsolatedNodes()

    alice = Entity(
        id="ent-1",
        entity_name="Alice",
        entity_type="Person",
        description="Alice",
        source_chunk_id=["chunk-1"],
    )
    bob = Entity(
        id="ent-2",
        entity_name="Bob",
        entity_type="Person",
        description="Bob",
        source_chunk_id=["chunk-2"],
    )

    entities, relations = await module.run([alice, bob], [])

    assert entities == []
    assert relations == []
