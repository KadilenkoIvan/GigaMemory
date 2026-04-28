"""
Bridge between DST memory pipeline and RAGU KnowledgeGraph.

Responsibilities:
  - Adapt RAGU's async API to the synchronous pipeline.
  - Translate DST triplet add/delete operations into RAGU Entity + Relation CRUD.
  - Wrap SentenceTransformer as a RAGU-compatible async Embedder.
  - Provide slot-filtered semantic search via LocalSearchEngine.
  - Expose embed_text_sync() for semantic deduplication in DSTManager.
  - Store TTL metadata in Entity/Relation attributes for visualization.

Slot encoding strategy
-----------------------
  Entity.entity_type  = slot name  (e.g. "FAMILY")
  Relation.slot       = slot name  (e.g. "FAMILY")

This makes пользователь(FAMILY) and пользователь(WORK) separate nodes,
naturally creating per-slot subgraphs while sharing the same NetworkX graph.
IDs are deterministic (MD5-based), so inserting the same entity twice is safe
(RAGU merges duplicate IDs automatically).

Triplet values are stored in lowercase with spaces (natural Russian), not UPPER_CASE_UNDERSCORE.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from sentence_transformers import SentenceTransformer

from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.graph.types import Entity, Relation
from ragu.models.embedder import Embedder
from ragu.search_engine.local_search import LocalSearchEngine
from ragu.search_engine.types import LocalSearchResult
from ragu.utils.ragu_utils import FLOATS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async Embedder adapter for SentenceTransformer
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder(Embedder):
    """
    Wraps a synchronous SentenceTransformer model to satisfy RAGU's async
    ``Embedder`` interface.  Encode calls are offloaded to a thread-pool
    executor so the event loop is not blocked.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        probe = self._model.encode([""], normalize_embeddings=True)
        self._dim: int = int(probe.shape[1])

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_text(self, text: str, **kwargs: Any) -> list[float] | FLOATS:
        loop = asyncio.get_event_loop()
        vector = await loop.run_in_executor(
            None,
            lambda: self._model.encode([text], normalize_embeddings=True)[0].tolist(),
        )
        return vector

    async def batch_embed_text(
        self,
        texts: list[str],
        desc: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]] | FLOATS:
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, normalize_embeddings=True).tolist(),
        )
        return vectors

    def encode_sync(self, text: str) -> List[float]:
        """Synchronous encode for use outside async context."""
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


# ---------------------------------------------------------------------------
# Internal mapping entry
# ---------------------------------------------------------------------------

@dataclass
class _RecordIds:
    subject_entity_id: str
    object_entity_id: str
    relation_id: str
    slot: str


# ---------------------------------------------------------------------------
# RaguGraphProcessor
# ---------------------------------------------------------------------------

class RaguGraphProcessor:
    """
    Synchronous façade for RAGU KnowledgeGraph that the DST pipeline calls.

    Because RAGU is fully async and the pipeline is sync, every public method
    runs the async operation in a dedicated thread with its own event loop via
    ``_run_in_new_loop()``.  This avoids conflicts with any outer event loop
    and is safe for the single-threaded synchronous pipeline.
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        local_search: LocalSearchEngine,
        embedder: Optional[SentenceTransformerEmbedder] = None,
    ) -> None:
        self.kg = knowledge_graph
        self.local_search = local_search
        self.embedder = embedder
        self._record_to_ids: dict[int, _RecordIds] = {}

    # ------------------------------------------------------------------
    # Embedding (for semantic dedup in DSTManager)
    # ------------------------------------------------------------------

    def embed_text_sync(self, text: str) -> List[float]:
        """Synchronously embed text. Used by DSTManager for semantic dedup."""
        if self.embedder is None:
            return []
        try:
            return self.embedder.encode_sync(text)
        except Exception as exc:
            logger.warning("embed_text_sync failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Public sync API
    # ------------------------------------------------------------------

    def upsert_triplet_deltas(self, deltas: List["GraphTripletDelta"]) -> None:
        """Insert new triplets into RAGU (sync wrapper)."""
        if not deltas:
            return
        _run_in_new_loop(self._async_upsert(deltas))

    def delete_triplet_deltas(self, deltas: List["GraphTripletDelete"]) -> None:
        """Delete deactivated triplets from RAGU (sync wrapper)."""
        if not deltas:
            return
        _run_in_new_loop(self._async_delete(deltas))

    def search_memory(
        self,
        query: str,
        slot_names: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[str]:
        """
        Semantic search over the knowledge graph.

        Returns a list of formatted strings like::

            [FAMILY] пользователь --есть жена--> жена пользователя (ttl: inf)

        Optionally filters results to ``slot_names`` when provided.
        """
        if not query:
            return []
        result: LocalSearchResult = _run_in_new_loop(
            self.local_search.a_search(query, top_k=top_k)
        )
        return self._format_result(result, slot_filter=set(slot_names) if slot_names else None)

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _async_upsert(self, deltas: List["GraphTripletDelta"]) -> None:
        for delta in deltas:
            await self._upsert_one(delta)

    async def _upsert_one(self, delta: "GraphTripletDelta") -> None:
        chunk_id = f"msg_{delta.dialogue_id}_{delta.step}"

        subj_entity = Entity(
            entity_name=delta.subject,
            entity_type=delta.slot,
            description=delta.subject,
            source_chunk_id=[chunk_id],
        )
        obj_entity = Entity(
            entity_name=delta.object,
            entity_type=delta.slot,
            description=delta.object,
            source_chunk_id=[chunk_id],
        )

        await self.kg.insert_entities([subj_entity, obj_entity])

        relation = Relation(
            subject_id=subj_entity.id,
            object_id=obj_entity.id,
            subject_name=delta.subject,
            object_name=delta.object,
            relation_type=delta.relation,
            description=f"{delta.subject} {delta.relation} {delta.object}",
            slot=delta.slot,
            source_chunk_id=[chunk_id],
        )

        # Store TTL in relation metadata via description extension and extra attrs
        # (RAGU's Relation doesn't have a custom field; we encode TTL in description)
        # TTL is appended so visualize_knowledge_graph.py can read it.
        relation_with_ttl = Relation(
            subject_id=subj_entity.id,
            object_id=obj_entity.id,
            subject_name=delta.subject,
            object_name=delta.object,
            relation_type=delta.relation,
            description=f"{delta.subject} {delta.relation} {delta.object} [ttl:{delta.ttl}]",
            slot=delta.slot,
            source_chunk_id=[chunk_id],
        )

        await self.kg.insert_relations([relation_with_ttl])

        self._record_to_ids[delta.record_id] = _RecordIds(
            subject_entity_id=subj_entity.id,
            object_entity_id=obj_entity.id,
            relation_id=relation_with_ttl.id,
            slot=delta.slot,
        )
        logger.debug(
            "RAGU upsert record_id=%d slot=%s [%s|%s|%s] ttl=%s",
            delta.record_id, delta.slot, delta.subject, delta.relation, delta.object, delta.ttl,
        )

    async def _async_delete(self, deltas: List["GraphTripletDelete"]) -> None:
        for delta in deltas:
            await self._delete_one(delta)

    async def _delete_one(self, delta: "GraphTripletDelete") -> None:
        ids = self._record_to_ids.get(delta.record_id)
        if ids is None:
            logger.warning(
                "RAGU delete: record_id=%d not found in mapping, skipping",
                delta.record_id,
            )
            return
        try:
            await self.kg.delete_relation(
                ids.subject_entity_id,
                ids.object_entity_id,
                ids.relation_id,
            )
        except Exception as exc:
            logger.warning("RAGU delete_relation failed record_id=%d: %s", delta.record_id, exc)
        finally:
            del self._record_to_ids[delta.record_id]
        logger.debug("RAGU delete record_id=%d relation_id=%s", delta.record_id, ids.relation_id)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ttl_from_description(description: str) -> Optional[str]:
        """Parse '[ttl:Xm]' annotation from relation description."""
        import re
        m = re.search(r'\[ttl:([^\]]+)\]', description or "")
        return m.group(1) if m else None

    @staticmethod
    def _format_result(
        result: LocalSearchResult,
        slot_filter: Optional[set] = None,
    ) -> List[str]:
        lines: List[str] = []
        seen: set[str] = set()

        for rel in result.relations:
            if slot_filter and rel.slot not in slot_filter:
                continue
            slot_tag = f"[{rel.slot}] " if rel.slot else ""
            ttl = RaguGraphProcessor._extract_ttl_from_description(getattr(rel, "description", ""))
            ttl_tag = f" (ttl: {ttl})" if ttl else ""
            line = f"{slot_tag}{rel.subject_name} --{rel.relation_type}--> {rel.object_name}{ttl_tag}"
            if line not in seen:
                seen.add(line)
                lines.append(line)

        if not lines:
            for ent in result.entities:
                if slot_filter and ent.entity_type not in slot_filter:
                    continue
                slot_tag = f"[{ent.entity_type}] " if ent.entity_type else ""
                line = f"{slot_tag}{ent.entity_name}: {ent.description}"
                if line not in seen:
                    seen.add(line)
                    lines.append(line)

        return lines


# ---------------------------------------------------------------------------
# Sync/async bridge
# ---------------------------------------------------------------------------

def _run_in_new_loop(coro) -> Any:
    """
    Execute *coro* in a brand-new event loop running in a daemon thread.
    """
    result_box: list = []
    exc_box: list = []

    def _target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_box.append(loop.run_until_complete(coro))
        except Exception as exc:
            exc_box.append(exc)
        finally:
            loop.close()

    t = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = t.submit(_target)
    future.result()
    t.shutdown(wait=False)

    if exc_box:
        raise exc_box[0]
    return result_box[0] if result_box else None


# ---------------------------------------------------------------------------
# Delta dataclasses (imported by DSTManager)
# ---------------------------------------------------------------------------

@dataclass
class GraphTripletDelta:
    """Payload for inserting a new triplet into the knowledge graph."""
    record_id: int
    dialogue_id: str
    step: int
    slot: str
    subject: str
    relation: str
    object: str
    ttl: str = "inf"


@dataclass
class GraphTripletDelete:
    """Payload for removing a triplet (deactivated by conflict resolution or TTL)."""
    record_id: int
    dialogue_id: str
    slot: str


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_ragu_processor(
    embedder_model: str = "all-MiniLM-L6-v2",
    storage_path: Optional[str] = None,
    language: str = "russian",
) -> Tuple[KnowledgeGraph, "RaguGraphProcessor"]:
    """
    Convenience factory: creates Embedder → KnowledgeGraph → LocalSearchEngine
    → RaguGraphProcessor in one call.
    """
    import os
    from pathlib import Path

    from ragu.common.global_parameters import Settings
    from ragu.graph.graph_builder_pipeline import BuilderArguments
    from ragu.graph.index import StorageArguments

    if storage_path is None:
        storage_path = str(Path(__file__).resolve().parent.parent / "ragu_storage")

    os.makedirs(storage_path, exist_ok=True)
    Settings.storage_folder = storage_path
    Settings.language = language

    embedder = SentenceTransformerEmbedder(embedder_model)

    builder_settings = BuilderArguments(
        make_community_summary=False,
        use_llm_summarization=False,
        use_clustering=False,
        remove_isolated_nodes=False,
        vectorize_chunks=False,
    )

    kg = KnowledgeGraph(
        llm=None,
        embedder=embedder,
        builder_settings=builder_settings,
        storage_settings=StorageArguments(),
        language=language,
    )

    local_search = LocalSearchEngine(
        llm=None,  # type: ignore[arg-type]
        knowledge_graph=kg,
        embedder=embedder,
        language=language,
    )

    processor = RaguGraphProcessor(
        knowledge_graph=kg,
        local_search=local_search,
        embedder=embedder,
    )
    return kg, processor
