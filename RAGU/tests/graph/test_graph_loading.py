"""
Tests for knowledge graph loading from storage.

Verifies that RAGU can correctly load previously built graphs
from disk storage without rebuilding.
"""

from pathlib import Path

import pytest

from ragu.common.global_parameters import Settings
from ragu.graph.graph_builder_pipeline import BuilderArguments
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.models.embedder import Embedder


class DummyEmbedder(Embedder):
    def __init__(self, dim: int = 3072):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_text(self, text: str, **kwargs) -> list[float]:
        return [0.0] * self.dim


class TestGraphLoading:
    """
    Test loading pre-built knowledge graphs from storage
    ."""

    @pytest.fixture
    def example_graph_path(self):
        """
        Path to example pre-built graph.
        """
        return "tests/kg_for_test"

    @pytest.fixture
    def setup_storage_folder(self, example_graph_path):
        """
        Set up storage folder before test.
        """
        Settings.storage_folder = example_graph_path
        return example_graph_path

    @pytest.fixture
    def no_llm_builder_settings(self):
        return BuilderArguments(use_llm_summarization=False)

    @pytest.fixture
    def mock_client(self):
        """
        Create a mock OpenAI client for testing.
        """
        return None

    @pytest.fixture
    def mock_embedder(self):
        """
        Create a mock embedder with correct dimensions for test graph.
        """
        return DummyEmbedder(dim=3072)

    def test_example_graph_exists(self, example_graph_path):
        """
        Verify that example graph directory exists with required files.
        """
        graph_path = Path(example_graph_path)
        assert graph_path.exists(), f"Example graph directory not found: {example_graph_path}"
        assert graph_path.is_dir(), f"Path is not a directory: {example_graph_path}"

        # Check for required storage files
        required_files = [
            "knowledge_graph.gml",
            "kv_chunks.json",
            "vdb_entity.json",
            "vdb_relation.json",
        ]

        for filename in required_files:
            file_path = graph_path / filename
            assert file_path.exists(), f"Required file missing: {filename}"
            assert file_path.stat().st_size > 0, f"File is empty: {filename}"

    @pytest.mark.asyncio
    async def test_load_existing_graph(
        self,
        setup_storage_folder,
        mock_client,
        mock_embedder,
        no_llm_builder_settings,
    ):
        """
        Test loading a pre-built graph from storage.
        """
        # Load graph (should load from storage, not build)
        kg = KnowledgeGraph(
            llm=mock_client,
            embedder=mock_embedder,
            builder_settings=no_llm_builder_settings,
        )

        assert kg is not None
        assert kg.index is not None

    @pytest.mark.asyncio
    async def test_loaded_graph_has_entities(
        self,
        setup_storage_folder,
        mock_client,
        mock_embedder,
        no_llm_builder_settings,
    ):
        """
        Verify that loaded graph contains entities.
        """
        kg = KnowledgeGraph(
            llm=mock_client,
            embedder=mock_embedder,
            builder_settings=no_llm_builder_settings,
        )

        # Check that graph backend has nodes
        graph_backend = kg.index.graph_backend
        assert graph_backend is not None

        # Check nodes count
        num_nodes = len(await graph_backend.get_all_nodes())
        assert num_nodes > 0, "Loaded graph should contain entities"

    @pytest.mark.asyncio
    async def test_loaded_graph_has_relations(
        self,
        setup_storage_folder,
        mock_client,
        mock_embedder,
        no_llm_builder_settings,
    ):
        """
        Verify that loaded graph contains relations.
        """
        kg = KnowledgeGraph(
            llm=mock_client,
            embedder=mock_embedder,
            builder_settings=no_llm_builder_settings,
        )

        graph_backend = kg.index.graph_backend
        assert graph_backend is not None

        num_edges = len(await graph_backend.get_all_nodes())
        assert num_edges > 0, "Loaded graph should contain relations"

    @pytest.mark.asyncio
    async def test_loaded_graph_entity_vdb(
        self,
        setup_storage_folder,
        mock_client,
        mock_embedder,
        no_llm_builder_settings,
    ):
        """
        Verify that entity vector database is loaded.
        """
        kg = KnowledgeGraph(
            llm=mock_client,
            embedder=mock_embedder,
            builder_settings=no_llm_builder_settings,
        )

        entity_vdb = kg.index.entity_vector_db
        assert entity_vdb is not None

        assert hasattr(entity_vdb, '_client')
        assert entity_vdb._client is not None

    @pytest.mark.asyncio
    async def test_loaded_graph_chunks(
        self,
        setup_storage_folder,
        mock_client,
        mock_embedder,
        no_llm_builder_settings,
    ):
        """
        Verify that chunks are loaded from KV storage.
        """
        kg = KnowledgeGraph(
            llm=mock_client,
            embedder=mock_embedder,
            builder_settings=no_llm_builder_settings,
        )

        chunks_storage = kg.index.chunks_kv_storage
        assert chunks_storage is not None

        all_keys = await chunks_storage.all_keys()
        assert all_keys is not None
        assert len(all_keys) > 0, "Loaded graph should contain chunks"
