# Tests

> Tests were made with Claude Code support.

## Structure

```
tests/
├── conftest.py                                 # Shared fixtures (mock embedder, Index factory, etc.)
├── chunker/
│   └── test_chunkers.py                        # SimpleChunker and SmartChunker splitting logic
├── common/
│   ├── test_batch_generator.py                 # Batch iteration utility
│   ├── test_cache.py                           # LRU / async cache helpers
│   └── test_env.py                             # Environment and settings resolution
├── embedder/
│   └── test_embedders.py                       # OpenAI and SentenceTransformers embedder wrappers
├── graph/
│   ├── test_artifacts_summarizer.py            # Entity and relation summarization / dedup
│   ├── test_builder_modules.py                 # GraphBuilderModule plugin system
│   ├── test_graph_loading.py                   # Persistence round-trip for KnowledgeGraph
│   └── test_graph_types.py                     # Entity / Relation dataclass behaviour
├── rerank/
│   ├── test_api_rerankers.py                   # API-based reranker clients
│   ├── test_base_reranker.py                   # BaseReranker interface contract
│   └── test_local_rerankers.py                 # Local model rerankers
├── search_engine/
│   ├── conftest.py                             # Fixtures that load kg_for_test/ into a real KnowledgeGraph
│   ├── test_global_search_engine.py            # GlobalSearchEngine (community summaries)
│   ├── test_local_search_engine.py             # LocalSearchEngine (entity embeddings)
│   └── test_naive_search_engine.py             # NaiveSearchEngine (chunk vector search + rerank)
├── storage/
│   ├── test_backend_batch_operations.py        # Batch upsert / delete across all backends
│   ├── test_index_crud.py                      # Index-level CRUD for entities, relations, chunks
│   ├── test_json_storage.py                    # JsonKVStorage adapter
│   ├── test_merge_logic.py                     # Entity / relation merge-on-conflict logic
│   ├── test_nano_vdb_storage.py                # NanoVectorDBStorage adapter
│   └── test_networkx_adapter.py                # NetworkXStorage graph backend
├── utils/
│   ├── test_ragu_utils.py                      # Hash IDs, file readers, misc helpers
│   └── test_token_truncation.py                # Token-aware text truncation
└── kg_for_test/                                # Pre-built knowledge graph fixtures (GML, JSON, VDB)
```

## Fixtures

- `tests/conftest.py` — project-wide fixtures such as mock embedders and temporary `Index` instances.
- `tests/search_engine/conftest.py` — loads `kg_for_test/` files into a real `KnowledgeGraph` (`real_kg`) used by search engine tests.
- `tests/kg_for_test/` — serialized graph, KV, and vector-DB files that provide a deterministic dataset for integration-level tests.
