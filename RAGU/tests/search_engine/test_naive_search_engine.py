from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragu.search_engine.naive_search import NaiveSearchEngine
from ragu.search_engine.types import NaiveSearchResult
from ragu.storage.types import EmbeddingHit


def _make_embedder_mock():
    return SimpleNamespace(embed_text=AsyncMock(return_value=[0.0] * 3))


@pytest.mark.asyncio
async def test_naive_search_rerank_and_rerank_top_k(real_kg, kg_fixture_ids):
    chunk_ids = kg_fixture_ids["chunk_ids"]
    real_kg.index.chunk_vector_db.query = AsyncMock(
        return_value=[
            EmbeddingHit(id=chunk_ids[0], distance=0.2),
            EmbeddingHit(id=chunk_ids[1], distance=0.8),
            EmbeddingHit(id="chunk-missing", distance=0.5),
        ]
    )

    reranker = SimpleNamespace(score=AsyncMock(return_value=[(1, 0.95), (0, 0.11)]))
    llm = SimpleNamespace(chat_completion=AsyncMock())

    engine = NaiveSearchEngine(
        llm=llm,
        knowledge_graph=real_kg,
        embedder=_make_embedder_mock(),
        reranker=reranker
    )
    result = await engine.a_search("query", top_k=3, rerank_top_k=1)

    assert isinstance(result, NaiveSearchResult)
    assert len(result.chunks) == 1
    assert result.chunks[0].id == chunk_ids[1]
    assert result.scores == [0.95]
    assert len(result.documents_id) == 1


@pytest.mark.asyncio
async def test_naive_search_empty_returns_empty_result(real_kg):
    real_kg.index.chunk_vector_db.query = AsyncMock(return_value=[])
    engine = NaiveSearchEngine(
        llm=SimpleNamespace(chat_completion=AsyncMock()),
        knowledge_graph=real_kg,
        embedder=_make_embedder_mock(),
    )

    result = await engine.a_search("query")
    assert result.chunks == []
    assert result.scores == []
    assert result.documents_id == []


@pytest.mark.asyncio
async def test_naive_query_uses_llm_response(monkeypatch):
    llm = SimpleNamespace(chat_completion=AsyncMock(return_value="naive-answer"))
    kg = SimpleNamespace(index=SimpleNamespace(chunk_vector_db=SimpleNamespace(query=AsyncMock(return_value=[]))))
    engine = NaiveSearchEngine(llm=llm, knowledge_graph=kg, embedder=_make_embedder_mock())
    engine.truncation = lambda s: s
    engine.a_search = AsyncMock(return_value=NaiveSearchResult())

    from ragu.search_engine import naive_search as naive_module
    monkeypatch.setattr(
        naive_module,
        "render",
        lambda messages, **kwargs: [SimpleNamespace(to_openai=lambda: [{"role": "user", "content": "prompt"}])],
    )
    monkeypatch.setattr(
        engine,
        "get_prompt",
        lambda _: SimpleNamespace(messages=[{"role": "user", "content": "{{query}}"}], pydantic_model=None),
    )

    result = await engine.a_query("question")
    assert result == "naive-answer"
