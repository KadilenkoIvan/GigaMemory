import json
from pathlib import Path

import pytest

from ragu.common.global_parameters import Settings
from ragu.models.embedder import Embedder
from ragu.graph.graph_builder_pipeline import BuilderArguments
from ragu.graph.knowledge_graph import KnowledgeGraph


class DummyEmbedder(Embedder):
    def __init__(self, dim: int = 3072):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_text(self, text: str, **kwargs) -> list[float]:
        return [0.001] * self.dim


@pytest.fixture
def real_kg():
    previous_storage = Settings.storage_folder
    Settings.storage_folder = "tests/kg_for_test"
    kg = KnowledgeGraph(
        llm=None,
        embedder=DummyEmbedder(dim=3072),
        builder_settings=BuilderArguments(use_llm_summarization=False),
    )
    yield kg
    Settings.storage_folder = previous_storage


@pytest.fixture
def kg_fixture_ids():
    chunks = json.loads(Path("tests/kg_for_test/kv_chunks.json").read_text(encoding="utf-8"))
    community_data = json.loads(Path("tests/kg_for_test/kv_community.json").read_text(encoding="utf-8"))
    first_community = next(iter(community_data.values()))
    return {
        "chunk_ids": list(chunks.keys()),
        "entity_ids": first_community.get("entity_ids", []),
    }
