from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragu.chunker.types import Chunk
from ragu.graph.types import Entity, Relation, CommunitySummary
from ragu.search_engine.local_search import LocalSearchEngine
from ragu.search_engine.types import LocalSearchResult
from ragu.storage.types import EmbeddingHit


def _make_embedder_mock():
    return SimpleNamespace(embed_text=AsyncMock(return_value=[0.0] * 3))


@pytest.mark.asyncio
async def test_local_search_collects_entities_relations_chunks_and_summaries(real_kg, kg_fixture_ids):
    entity_ids = kg_fixture_ids["entity_ids"]
    real_kg.index.entity_vector_db.query = AsyncMock(
        return_value=[
            EmbeddingHit(id=entity_ids[0], distance=0.9),
            EmbeddingHit(id=entity_ids[1], distance=0.8),
            EmbeddingHit(id="ent-missing", distance=0.7),
        ]
    )
    engine = LocalSearchEngine(
        llm=SimpleNamespace(chat_completion=AsyncMock()),
        knowledge_graph=real_kg,
        embedder=_make_embedder_mock(),
    )

    result = await engine.a_search("query", top_k=3)

    assert isinstance(result, LocalSearchResult)
    assert [e.id for e in result.entities] == entity_ids[:2]
    assert isinstance(result.relations, list)
    assert isinstance(result.chunks, list)
    assert isinstance(result.summaries, list)
    assert isinstance(result.documents_id, list)


@pytest.mark.asyncio
async def test_local_search_reranks_entities_relations_summaries_and_chunks(monkeypatch, real_kg):
    llm = SimpleNamespace(chat_completion=AsyncMock())
    entity_a = Entity(entity_name="Alpha", entity_type="Person", description="First entity", source_chunk_id=["chunk-a"], documents_id=["doc-a"])
    entity_b = Entity(entity_name="Beta", entity_type="Place", description="Second entity", source_chunk_id=["chunk-b"], documents_id=["doc-b"])
    relation_a = Relation(
        subject_id=entity_a.id,
        object_id=entity_b.id,
        subject_name=entity_a.entity_name,
        object_name=entity_b.entity_name,
        relation_type="knows",
        description="Alpha knows Beta",
    )
    relation_b = Relation(
        subject_id=entity_b.id,
        object_id=entity_a.id,
        subject_name=entity_b.entity_name,
        object_name=entity_a.entity_name,
        relation_type="visited",
        description="Beta visited Alpha",
    )
    chunk_a = Chunk(content="chunk alpha", chunk_order_idx=0, doc_id="doc-a")
    chunk_b = Chunk(content="chunk beta", chunk_order_idx=1, doc_id="doc-b")
    summaries = [CommunitySummary(summary="summary alpha", id="123"), CommunitySummary(summary="summary beta", id="345")]

    real_kg.index.entity_vector_db.query = AsyncMock(
        return_value=[
            EmbeddingHit(id=entity_a.id, distance=0.9),
            EmbeddingHit(id=entity_b.id, distance=0.8),
        ]
    )
    real_kg.index.get_entities = AsyncMock(return_value=[entity_a, entity_b])

    reranker = SimpleNamespace(
        score=AsyncMock(
            side_effect=[
                [(1, 0.9), (0, 0.1)],
                [(1, 0.8), (0, 0.2)],
                [(1, 0.7), (0, 0.3)],
                [(1, 0.6), (0, 0.4)],
            ]
        )
    )

    from ragu.search_engine import local_search as local_module
    monkeypatch.setattr(local_module, "_find_most_related_edges_from_entities", AsyncMock(return_value=[relation_a, relation_b]))
    monkeypatch.setattr(local_module, "_find_most_related_text_unit_from_entities", AsyncMock(return_value=[chunk_a, chunk_b]))
    monkeypatch.setattr(local_module, "_find_most_related_community_from_entities", AsyncMock(return_value=summaries))
    monkeypatch.setattr(local_module, "_find_documents_id", AsyncMock(return_value=["doc-b", "doc-a"]))

    engine = LocalSearchEngine(
        llm=llm,
        knowledge_graph=real_kg,
        embedder=_make_embedder_mock(),
        reranker=reranker,
    )

    result = await engine.a_search("query", top_k=2)

    assert [entity.id for entity in result.entities] == [entity_b.id, entity_a.id]
    assert [relation.id for relation in result.relations] == [relation_b.id, relation_a.id]
    assert [c.summary for c in result.summaries] == ["summary beta", "summary alpha"]
    assert [chunk.id for chunk in result.chunks] == [chunk_b.id, chunk_a.id]
    assert result.documents_id == ["doc-b", "doc-a"]


@pytest.mark.asyncio
async def test_local_query_returns_raw_result_when_no_response_attr(monkeypatch, real_kg):
    llm = SimpleNamespace(chat_completion=AsyncMock(return_value="raw-result"))
    engine = LocalSearchEngine(llm=llm, knowledge_graph=real_kg, embedder=_make_embedder_mock())
    engine.truncation = lambda s: s
    engine.a_search = AsyncMock(return_value=LocalSearchResult())

    from ragu.search_engine import local_search as local_module
    monkeypatch.setattr(
        local_module,
        "render",
        lambda messages, **kwargs: [SimpleNamespace(to_openai=lambda: [{"role": "user", "content": "prompt"}])],
    )
    monkeypatch.setattr(
        engine,
        "get_prompt",
        lambda _: SimpleNamespace(messages=[{"role": "user", "content": "{{query}}"}], pydantic_model=None),
    )

    result = await engine.a_query("question")
    assert result == "raw-result"
