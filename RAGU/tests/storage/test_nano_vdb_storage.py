from __future__ import annotations

from typing import Dict, List, Optional

import pytest

from ragu.models.embedder import Embedder
from ragu.storage.types import Embedding, EmbeddingHit
from ragu.storage.vdb_storage_adapters.nano_vdb import NanoVectorDBStorage


class DummyEmbedder(Embedder):
    def __init__(self, dim: int, vector_by_text: Dict[str, Optional[List[float]]]):
        self._dim = dim
        self.vector_by_text = vector_by_text

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_text(self, text: str, **kwargs) -> list[float]:
        vector = self.vector_by_text.get(text)
        assert vector is not None
        return vector


def _make_embeddings(data: Dict[str, dict], embedder: DummyEmbedder) -> List[Embedding]:
    """Helper to build Embedding objects from old-style dict format."""
    result = []
    for record_id, payload in data.items():
        content = payload.get("content", "")
        vector = embedder.vector_by_text.get(content)
        metadata = {k: v for k, v in payload.items() if k != "content"}
        result.append(Embedding(id=record_id, vector=vector, metadata=metadata))
    return result


@pytest.mark.asyncio
async def test_upsert_and_query_returns_expected_fields(tmp_path):
    embedder = DummyEmbedder(
        dim=3,
        vector_by_text={
            "alpha": [1.0, 0.0, 0.0],
            "beta": [0.0, 1.0, 0.0],
            "query-alpha": [1.0, 0.0, 0.0],
        },
    )
    storage_file = tmp_path / "vdb.json"
    vdb = NanoVectorDBStorage(
        embedding_dim=embedder.dim,
        filename=str(storage_file),
        cosine_threshold=0.2,
    )

    embeddings = _make_embeddings(
        {"id-alpha": {"content": "alpha", "tag": "A"}, "id-beta": {"content": "beta", "tag": "B"}},
        embedder,
    )
    await vdb.upsert(embeddings)

    query_vector = embedder.vector_by_text["query-alpha"]
    results = await vdb.query(Embedding(vector=query_vector), top_k=10)

    assert len(results) == 1
    assert isinstance(results[0], EmbeddingHit)
    assert results[0].id == "id-alpha"
    assert "distance" in results[0].__dataclass_fields__ or hasattr(results[0], "distance")
    assert results[0].metadata["tag"] == "A"


@pytest.mark.asyncio
async def test_upsert_empty_returns_empty_list(tmp_path):
    storage_file = tmp_path / "vdb.json"
    vdb = NanoVectorDBStorage(embedding_dim=3, filename=str(storage_file))

    inserted = await vdb.upsert([])

    assert inserted is None


@pytest.mark.asyncio
async def test_upsert_rejects_none_embeddings(tmp_path):
    embedder = DummyEmbedder(
        dim=3,
        vector_by_text={
            "keep": [1.0, 0.0, 0.0],
            "drop": None,
            "query-keep": [1.0, 0.0, 0.0],
        },
    )
    storage_file = tmp_path / "vdb.json"
    vdb = NanoVectorDBStorage(embedding_dim=embedder.dim, filename=str(storage_file), cosine_threshold=0.0)

    embeddings = _make_embeddings(
        {"id-keep": {"content": "keep"}, "id-drop": {"content": "drop"}},
        embedder,
    )
    with pytest.raises(AssertionError):
        await vdb.upsert(embeddings)


@pytest.mark.asyncio
async def test_delete_and_persistence_round_trip(tmp_path):
    embedder = DummyEmbedder(
        dim=3,
        vector_by_text={
            "persist": [1.0, 0.0, 0.0],
            "query": [1.0, 0.0, 0.0],
        },
    )
    storage_file = tmp_path / "vdb.json"
    vdb = NanoVectorDBStorage(embedding_dim=embedder.dim, filename=str(storage_file), cosine_threshold=0.0)

    embeddings = _make_embeddings({"id-persist": {"content": "persist"}}, embedder)
    await vdb.upsert(embeddings)
    await vdb.index_done_callback()

    reloaded = NanoVectorDBStorage(embedding_dim=embedder.dim, filename=str(storage_file), cosine_threshold=0.0)
    query_vector = embedder.vector_by_text["query"]
    loaded_results = await reloaded.query(Embedding(vector=query_vector), top_k=10)
    assert any(r.id == "id-persist" for r in loaded_results)

    await reloaded.delete([])
    await reloaded.delete(["id-persist"])
    post_delete_results = await reloaded.query(Embedding(vector=query_vector), top_k=10)
    assert all(r.id != "id-persist" for r in post_delete_results)
