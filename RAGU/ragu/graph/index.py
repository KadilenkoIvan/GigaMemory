from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    cast
)

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import DEFAULT_FILENAMES
from ragu.common.global_parameters import Settings
from ragu.graph.types import (
    ClusterInfo,
    Entity,
    Relation,
    Community,
    CommunitySummary
)
from ragu.models.embedder import Embedder
from ragu.storage.base_storage import (
    BaseKVStorage,
    BaseVectorStorage,
    BaseGraphStorage,
    EdgeSpec,
)
from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage
from ragu.storage.kv_storage_adapters.json_storage import JsonKVStorage
from ragu.storage.types import Embedding
from ragu.storage.vdb_storage_adapters.nano_vdb import NanoVectorDBStorage


@dataclass
class StorageArguments:
    """
    Configuration for Index storage backends.

    :param graph_backend_storage: Storage backend class for graph structure (nodes/edges).
    :param kv_storage_type: Storage backend class for key-value data (chunks, communities, summaries).
    :param vdb_storage_type: Storage backend class for vector embeddings (entities, relations, chunks).
    :param chunks_kv_storage_kwargs: Additional kwargs passed to KV storage for text chunks.
    :param summary_kv_storage_kwargs: Additional kwargs passed to KV storage for community summaries.
    :param communities_kv_storage_kwargs: Additional kwargs passed to KV storage for community metadata.
    :param vdb_storage_kwargs: Additional kwargs passed to vector database instances.
    :param graph_storage_kwargs: Additional kwargs passed to graph backend storage.
    """
    graph_backend_storage: Type[BaseGraphStorage] = NetworkXStorage
    kv_storage_type: Type[BaseKVStorage[Any]] = JsonKVStorage
    vdb_storage_type: Type[BaseVectorStorage] = NanoVectorDBStorage

    chunks_kv_storage_kwargs: Dict[str, Any] = field(default_factory=dict[str, Any])
    summary_kv_storage_kwargs: Dict[str, Any] = field(default_factory=dict[str, Any])
    communities_kv_storage_kwargs: Dict[str, Any] = field(default_factory=dict[str, Any])
    vdb_storage_kwargs: Dict[str, Any] = field(default_factory=dict[str, Any])
    graph_storage_kwargs: Dict[str, Any] = field(default_factory=dict[str, Any])


class Index:
    """
    Manages all storage operations for a knowledge graph.

    Coordinates three storage backends (graph, vector DB, KV) and provides
    batch CRUD operations with cascading deletes and duplicate merging.
    """

    def __init__(
            self,
            embedder: Embedder,  # cannot be optional, since calls .dim in __init__
            arguments: StorageArguments,
    ):
        """
        Initialize storage backends and in-memory reverse indexes.

        :param embedder: Embedder used for text-to-vector conversion.
        :param arguments: Configuration for storage backend implementations.
        """
        Settings.init_storage_folder()
        storage_folder: str = Settings.storage_folder

        self.embedder = embedder

        # Reverse indexes for cascade operations
        self._chunk_to_entities: Dict[str, Set[str]] = defaultdict(set)
        self._chunk_to_relations: Dict[str, Set[str]] = defaultdict(set)

        summary_kv_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["community_summary_kv_storage_name"],
            arguments.summary_kv_storage_kwargs,
        )
        community_kv_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["community_kv_storage_name"],
            arguments.communities_kv_storage_kwargs,
        )
        chunks_kv_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["chunks_kv_storage_name"],
            arguments.chunks_kv_storage_kwargs,
        )
        entity_vdb_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["entity_vdb_name"],
            arguments.vdb_storage_kwargs,
        )
        relation_vdb_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["relation_vdb_name"],
            arguments.vdb_storage_kwargs,
        )
        chunk_vdb_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["chunk_vdb_name"],
            arguments.vdb_storage_kwargs,
        )
        graph_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["knowledge_graph_storage_name"],
            arguments.graph_storage_kwargs,
        )

        # Key-value storages
        self.chunks_kv_storage = arguments.kv_storage_type(**chunks_kv_kwargs)
        self.community_summary_kv_storage = arguments.kv_storage_type(**summary_kv_kwargs)
        self.community_kv_storage = arguments.kv_storage_type(**community_kv_kwargs)

        embedding_dim_from_embedder = embedder.dim

        dimensions_from_kwargs = [
            storage_kwargs.get("embedding_dim") for storage_kwargs in [
                entity_vdb_kwargs,
                relation_vdb_kwargs,
                chunk_vdb_kwargs
            ] if storage_kwargs.get("embedding_dim")]

        number_of_dimensions = len(dimensions_from_kwargs)
        if number_of_dimensions > 1:
            raise ValueError(f"Dimension mismatch in vdb kwargs: {dimensions_from_kwargs}")
        if number_of_dimensions == 1:
            if dimensions_from_kwargs[0] != embedding_dim_from_embedder:
                raise ValueError(f"Dimension mismatch in vdb kwargs and embedder setup: "
                                 f"{dimensions_from_kwargs[0]} and {embedding_dim_from_embedder}")

        resolved_dim = embedding_dim_from_embedder

        entity_vdb_kwargs["embedding_dim"] = resolved_dim
        relation_vdb_kwargs["embedding_dim"] = resolved_dim
        chunk_vdb_kwargs["embedding_dim"] = resolved_dim

        # Vector storages
        self.entity_vector_db = arguments.vdb_storage_type(**entity_vdb_kwargs)
        self.relation_vector_db = arguments.vdb_storage_type(**relation_vdb_kwargs)
        self.chunk_vector_db = arguments.vdb_storage_type(**chunk_vdb_kwargs)

        # Graph storage
        self.graph_backend = arguments.graph_backend_storage(
            node_cls=Entity,
            edge_cls=Relation,
            **graph_kwargs
        )

    async def insert_entities(self, entities: List[Entity]) -> "Index":
        """
        Insert entities into graph and vector DB.

        Duplicate IDs in the incoming batch are merged. If an entity with the
        same ID already exists, incoming and existing values are merged.

        :param entities: Entities to insert.
        :return: Self for method chaining.
        """
        if not entities:
            return self

        incoming_by_id: Dict[str, List[Entity]] = defaultdict(list)
        for entity in entities:
            assert entity.id  # according to incoming_by_id type hint, should be not None
            incoming_by_id[entity.id].append(entity)

        existing_entities = await self.graph_backend.get_nodes(list(incoming_by_id.keys()))
        existing_by_id: Dict[str, Entity] = {}
        for e in existing_entities:
            if e is not None:
                assert e.id  # according to existing_by_id type hint, should be not None
                existing_by_id[e.id] = e

        entities_to_insert: List[Entity] = []
        for entity_id, incoming_group in incoming_by_id.items():
            merged_group = list(incoming_group)
            existing = existing_by_id.get(entity_id)
            if existing is not None:
                merged_group.append(existing)

            if len(merged_group) > 1:
                entities_to_insert.extend(self._merge_entities({entity_id: merged_group}))
            else:
                entities_to_insert.extend(incoming_group)

        await self.graph_backend.upsert_nodes(entities_to_insert)
        vdb_data = {
            e.id: {
                "entity_name": e.entity_name,
                "content": f"{e.entity_name} - {e.description}",
            }
            for e in entities_to_insert
        }
        embeddings = await self._build_embeddings(vdb_data, tqdm_title="Entities vectorization")
        await self.entity_vector_db.upsert(embeddings)

        await self.graph_backend.index_done_callback()
        await self.entity_vector_db.index_done_callback()
        await self._update_reverse_indexes(entities=entities_to_insert)
        return self

    async def update_entities(self, entities: List[Entity]) -> "Index":
        """
        Update entities by ID using replace semantics.

        Existing entities are replaced by incoming payloads. No merge with
        previous values is performed.

        :param entities: Entities to update.
        :return: Self for method chaining.
        :raises ValueError: If entity IDs are missing/duplicated in request or absent in storage.
        """
        if not entities:
            return self

        incoming_by_id: Dict[str, List[Entity]] = defaultdict(list)
        for entity in entities:
            assert entity.id
            incoming_by_id[entity.id].append(entity)

        duplicate_ids = [entity_id for entity_id, group in incoming_by_id.items() if len(group) > 1]
        if duplicate_ids:
            raise ValueError(f"Cannot update duplicated entity IDs in one request: {duplicate_ids}")

        entity_ids = list(incoming_by_id.keys())
        existing_entities = await self.graph_backend.get_nodes(entity_ids)
        missing_ids = [entity_id for entity_id, existing in zip(entity_ids, existing_entities) if existing is None]
        if missing_ids:
            raise ValueError(f"Cannot update non-existent entities: {missing_ids}")

        entities_to_update = [group[0] for group in incoming_by_id.values()]

        await self.graph_backend.upsert_nodes(entities_to_update)
        vdb_data = {
            e.id: {
                "entity_name": e.entity_name,
                "content": f"{e.entity_name} - {e.description}",
            }
            for e in entities_to_update
        }
        embeddings = await self._build_embeddings(vdb_data, tqdm_title="Entities vectorization")
        await self.entity_vector_db.upsert(embeddings)

        await self.graph_backend.index_done_callback()
        await self.entity_vector_db.index_done_callback()

        await self._update_reverse_indexes(
            deleted_entity_ids=entity_ids,
            entities=entities_to_update,
        )
        return self

    async def insert_relations(self, relations: List[Relation]) -> "Index":
        """
        Insert relations into graph and vector DB.

        Duplicate IDs in the incoming batch are merged. If a relation with the
        same ID already exists, incoming and existing values are merged.

        :param relations: Relations to insert.
        :return: Self for method chaining.
        :raises ValueError: If referenced entities don't exist.
        """
        if not relations:
            return self

        for relation in relations:
            if not relation.id:
                raise ValueError("Cannot insert relation without id")

        await self._validate_relation_endpoints_exist(relations)

        incoming_by_id: Dict[str, List[Relation]] = defaultdict(list)
        for relation in relations:
            assert relation.id
            incoming_by_id[relation.id].append(relation)

        existing_relation_groups = await self._get_existing_relations_grouped_by_id(set(incoming_by_id.keys()))
        existing_relation_ids = list(existing_relation_groups.keys())

        relations_to_insert: List[Relation] = []
        for relation_id, incoming_group in incoming_by_id.items():
            merged_group = list(incoming_group)
            merged_group.extend(existing_relation_groups.get(relation_id, []))

            if len(merged_group) > 1:
                relations_to_insert.extend(self._merge_relations({relation_id: merged_group}))
            else:
                relations_to_insert.extend(incoming_group)

        delete_specs: List[EdgeSpec] = [
            (relation.subject_id, relation.object_id, relation.id)
            for relations_group in existing_relation_groups.values()
            for relation in relations_group
            if relation.id
        ]
        if delete_specs:
            await self.graph_backend.delete_edges(delete_specs)

        await self.graph_backend.upsert_edges(relations_to_insert)

        vdb_data = {
            r.id: {
                "subject": r.subject_id,
                "object": r.object_id,
                "content": r.description,
            }
            for r in relations_to_insert
        }
        if existing_relation_ids:
            await self.relation_vector_db.delete(existing_relation_ids)
        embeddings = await self._build_embeddings(vdb_data, tqdm_title="Relations vectorization")
        await self.relation_vector_db.upsert(embeddings)

        await self.graph_backend.index_done_callback()
        await self.relation_vector_db.index_done_callback()
        await self._update_reverse_indexes(
            deleted_relation_ids=existing_relation_ids,
            relations=relations_to_insert,
        )
        return self

    async def update_relations(self, relations: List[Relation]) -> "Index":
        """
        Update relations by ID using replace semantics.

        Existing relations are replaced by incoming payloads. No merge with
        previous values is performed.

        :param relations: Relations to update.
        :return: Self for method chaining.
        :raises ValueError: If relation IDs are missing/duplicated in request,
            IDs are absent in storage, or referenced entities don't exist.
        """
        if not relations:
            return self

        incoming_by_id: Dict[str, List[Relation]] = defaultdict(list)
        for relation in relations:
            if not relation.id:
                raise ValueError("Cannot update relation without id")
            incoming_by_id[relation.id].append(relation)

        duplicate_ids = [relation_id for relation_id, group in incoming_by_id.items() if len(group) > 1]
        if duplicate_ids:
            raise ValueError(f"Cannot update duplicated relation IDs in one request: {duplicate_ids}")

        relation_ids = list(incoming_by_id.keys())
        existing_relation_groups = await self._get_existing_relations_grouped_by_id(set(relation_ids))
        missing_ids = [relation_id for relation_id in relation_ids if relation_id not in existing_relation_groups]
        if missing_ids:
            raise ValueError(f"Cannot update non-existent relations: {missing_ids}")

        relations_to_update = [group[0] for group in incoming_by_id.values()]
        await self._validate_relation_endpoints_exist(relations_to_update)

        delete_specs: List[EdgeSpec] = [
            (relation.subject_id, relation.object_id, relation.id)
            for relations_group in existing_relation_groups.values()
            for relation in relations_group
            if relation.id
        ]
        if delete_specs:
            await self.graph_backend.delete_edges(delete_specs)

        await self.graph_backend.upsert_edges(relations_to_update)

        vdb_data = {
            r.id: {
                "subject": r.subject_id,
                "object": r.object_id,
                "content": r.description,
            }
            for r in relations_to_update
        }
        await self.relation_vector_db.delete(relation_ids)
        embeddings = await self._build_embeddings(vdb_data, tqdm_title="Relations vectorization")
        await self.relation_vector_db.upsert(embeddings)

        await self.graph_backend.index_done_callback()
        await self.relation_vector_db.index_done_callback()
        await self._update_reverse_indexes(
            deleted_relation_ids=relation_ids,
            relations=relations_to_update,
        )
        return self

    async def upsert_chunks(self, chunks: List[Chunk]) -> "Index":
        """
        Insert or update chunks into KV storage (and optionally vector DB).

        :param chunks: Chunks to upsert.
        :return: Self for method chaining.
        """
        if not chunks:
            return self

        # Store in KV
        kv_data: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            chunk_dict = asdict(chunk)
            chunk_id = cast(str, chunk_dict.pop("id"))
            kv_data[chunk_id] = chunk_dict

        await self.chunks_kv_storage.upsert(kv_data)

        vdb_data = {
            c.id: {"content": c.content, "doc_id": c.doc_id}
            for c in chunks
        }
        embeddings = await self._build_embeddings(vdb_data, tqdm_title="Chunks vectorization")
        await self.chunk_vector_db.upsert(embeddings)
        await self.chunk_vector_db.index_done_callback()

        await self.chunks_kv_storage.index_done_callback()
        return self

    async def reindex_cluster_ids(
        self,
        entities: List[Entity],
        communities: List[Community],
        summaries: Optional[List[CommunitySummary]] = None,
    ) -> tuple[List[Entity], List[Community], List[CommunitySummary]]:
        """
        Remap cluster IDs to be globally unique per level across indexing runs.

        Levels are preserved to keep level-based filtering intact.

        :param entities: Entities whose cluster memberships should be remapped.
        :param communities: Newly generated communities with local cluster IDs.
        :param summaries: Optional summaries linked to community IDs.
        :return: Tuple with remapped entities, remapped communities, and remapped summaries.
        """
        if not communities:
            return entities, communities, summaries or []

        existing_keys = await self.community_kv_storage.all_keys()
        existing_data = await self.community_kv_storage.get_by_ids(existing_keys) if existing_keys else []

        max_cluster_id_by_level: Dict[int, int] = defaultdict(lambda: -1)
        for row in existing_data:
            if not row:
                continue
            try:
                level = int(row.get("level"))
                cluster_id = int(row.get("cluster_id"))
            except (TypeError, ValueError):
                continue
            if cluster_id > max_cluster_id_by_level[level]:
                max_cluster_id_by_level[level] = cluster_id

        local_ids_by_level: Dict[int, List[int]] = defaultdict(list)
        for community in communities:
            level = int(community.level)
            cluster_id = int(community.cluster_id)
            local_ids_by_level[level].append(cluster_id)

        local_to_global: Dict[Tuple[int, int], int] = {}
        for level, local_ids in local_ids_by_level.items():
            for local_cluster_id in sorted(set(local_ids)):
                max_cluster_id_by_level[level] += 1
                local_to_global[(level, local_cluster_id)] = max_cluster_id_by_level[level]

        old_to_new_community_id: Dict[str, str] = {}
        remapped_communities: List[Community] = []
        for community in communities:
            level = int(community.level)
            local_cluster_id = int(community.cluster_id)
            global_cluster_id = local_to_global[(level, local_cluster_id)]

            remapped_community = Community(
                level=level,
                cluster_id=global_cluster_id,
                entities=community.entities,
                relations=community.relations,
            )
            if community.id:
                old_to_new_community_id[str(community.id)] = str(remapped_community.id)
            remapped_communities.append(remapped_community)

        valid_cluster_pairs: Set[Tuple[int, int]] = {
            (int(community.level), int(community.cluster_id))
            for community in remapped_communities
        }

        for entity in entities:
            remapped_memberships: List[ClusterInfo] = []
            seen_memberships: Set[Tuple[int, int]] = set()
            for membership in entity.clusters:
                assert isinstance(membership, dict), f'What is membership? {membership}'
                try:
                    level = membership['level']
                    local_cluster_id = membership["cluster_id"]
                except (TypeError, ValueError):
                    continue

                global_cluster_id = local_to_global.get((level, local_cluster_id), local_cluster_id)
                if (level, global_cluster_id) not in valid_cluster_pairs:
                    continue

                membership_key = (level, global_cluster_id)
                if membership_key in seen_memberships:
                    continue

                seen_memberships.add(membership_key)
                remapped_memberships.append({
                    "level": level,
                    "cluster_id": global_cluster_id,
                })
            entity.clusters = remapped_memberships

        remapped_summaries: List[CommunitySummary] = []
        if summaries:
            for summary in summaries:
                assert summary is not None, 'Why summary is None?'
                new_summary_id = old_to_new_community_id.get(str(summary.id), summary.id)
                remapped_summaries.append(
                    CommunitySummary(
                        summary=summary.summary,
                        id=new_summary_id,
                    )
                )

        return entities, remapped_communities, remapped_summaries

    async def upsert_communities(self, communities: List[Community]) -> "Index":
        """
        Insert or update communities into KV storage.

        :param communities: Communities to upsert.
        :return: Self for method chaining.
        """
        if not communities:
            return self

        kv_data: dict[str, Any] = {
            c.id: {
                "level": c.level,
                "cluster_id": c.cluster_id,
                "entity_ids": sorted({e.id for e in c.entities}),
                "relation_ids": sorted({r.id for r in c.relations}),
            }
            for c in communities
        }
        await self.community_kv_storage.upsert(kv_data)
        await self.community_kv_storage.index_done_callback()
        return self

    async def upsert_summaries(self, summaries: List[CommunitySummary]) -> "Index":
        """
        Insert or update community summaries into KV storage.

        :param summaries: Summaries to upsert.
        :return: Self for method chaining.
        """
        if not summaries:
            return self

        kv_data = {s.id: s.summary for s in summaries}
        await self.community_summary_kv_storage.upsert(kv_data)
        await self.community_summary_kv_storage.index_done_callback()
        return self

    async def delete_entities(self, entity_ids: List[str]) -> "Index":
        """
        Delete entities from graph and vector DB.

        All relations connected to the deleted
        entities are also removed from the relation vector DB.

        :param entity_ids: IDs of entities to delete.
        :return: Self for method chaining.
        """
        if not entity_ids:
            return self

        relations_by_node = await self.graph_backend.get_all_edges_for_nodes(entity_ids)
        relation_ids = self._unique_relation_ids_from_grouped(relations_by_node)

        await self.graph_backend.delete_nodes(entity_ids)
        await self.entity_vector_db.delete(entity_ids)

        await self.relation_vector_db.delete(relation_ids)
        await self.relation_vector_db.index_done_callback()

        await self.graph_backend.index_done_callback()
        await self.entity_vector_db.index_done_callback()
        await self._update_reverse_indexes(
            deleted_entity_ids=entity_ids,
            deleted_relation_ids=relation_ids,
        )
        return self

    async def delete_relations(self, edge_specs: List[EdgeSpec]) -> "Index":
        """
        Delete relations from graph and vector DB.

        :param edge_specs: List of edge specs ``(subject_id, object_id, relation_id)``.
        :return: Self for method chaining.
        """
        if not edge_specs:
            return self

        # Fetch relation IDs for vector DB deletion before removing from graph
        relations = await self.graph_backend.get_edges(edge_specs)
        found_relation_ids = [r.id for r in relations if r is not None and r.id]

        await self.graph_backend.delete_edges(edge_specs)

        if found_relation_ids:
            await self.relation_vector_db.delete(found_relation_ids)

        await self.graph_backend.index_done_callback()
        await self.relation_vector_db.index_done_callback()
        await self._update_reverse_indexes(deleted_relation_ids=found_relation_ids)
        return self

    async def delete_chunks(self, chunk_ids: List[str]) -> "Index":
        """
        Delete chunks from KV and vector storage.

        :param chunk_ids: IDs of chunks to delete.
        :return: Self for method chaining.
        """
        if not chunk_ids:
            return self

        affected_entities = await self._find_entities_by_chunk_ids(chunk_ids)
        entity_ids = [e.id for e in affected_entities]
        relation_ids = []

        if entity_ids:
            # Collect relation IDs before graph cascade
            relations_by_node = await self.graph_backend.get_all_edges_for_nodes(entity_ids)
            relation_ids = self._unique_relation_ids_from_grouped(relations_by_node)

            await self.graph_backend.delete_nodes(entity_ids)
            await self.entity_vector_db.delete(entity_ids)

            if relation_ids:
                await self.relation_vector_db.delete(relation_ids)

        await self.chunks_kv_storage.delete(chunk_ids)
        await self.chunk_vector_db.delete(chunk_ids)

        await self.chunks_kv_storage.index_done_callback()
        await self.chunk_vector_db.index_done_callback()
        if entity_ids:
            await self.graph_backend.index_done_callback()
            await self.entity_vector_db.index_done_callback()
        if relation_ids:
            await self.relation_vector_db.index_done_callback()
        await self._update_reverse_indexes(
            deleted_chunk_ids=chunk_ids,
            deleted_entity_ids=entity_ids,
            deleted_relation_ids=relation_ids,
        )
        return self

    async def delete_communities(self, community_ids: List[str]) -> "Index":
        """
        Delete communities and their summaries from KV storage.

        :param community_ids: IDs of communities to delete.
        :return: Self for method chaining.
        """
        if not community_ids:
            return self

        await self.community_kv_storage.delete(community_ids)
        await self.community_summary_kv_storage.delete(community_ids)

        await self.community_kv_storage.index_done_callback()
        await self.community_summary_kv_storage.index_done_callback()
        return self

    async def get_entities(self, entity_ids: List[str]) -> List[Optional[Entity]]:
        """
        Retrieve entities by their IDs.

        :param entity_ids: Entity IDs to fetch.
        :return: List of entities (``None`` for missing).
        """
        return await self.graph_backend.get_nodes(entity_ids)

    async def get_relations(self, edge_specs: List[EdgeSpec]) -> List[Optional[Relation]]:
        """
        Retrieve relations by edge specs.

        :param edge_specs: List of edge specs ``(subject_id, object_id, relation_id)``.
        :return: List of relations (``None`` for missing).
        """
        return await self.graph_backend.get_edges(edge_specs)

    async def get_chunks(self, chunk_ids: List[str]) -> List[Optional[Chunk]]:
        """
        Retrieve chunks by their IDs.

        :param chunk_ids: Chunk IDs to fetch.
        :return: List of chunks (``None`` for missing).
        """
        chunk_dicts = await self.chunks_kv_storage.get_by_ids(chunk_ids)
        result: list[Chunk | None] = []
        for chunk_dict in chunk_dicts:
            if chunk_dict is None:
                result.append(None)
            else:
                result.append(Chunk(**chunk_dict))
        return result

    async def get_communities(self, community_ids: List[str]) -> List[Optional[Community]]:
        """
        Retrieve communities by their IDs, reconstructing from stored metadata.

        :param community_ids: Community IDs to fetch.
        :return: List of communities (``None`` for missing).
        """
        community_dicts = await self.community_kv_storage.get_by_ids(community_ids)
        communities: list[Community | None] = []

        for community_id, community_dict in zip(community_ids, community_dicts):
            if community_dict is None:
                communities.append(None)
                continue

            entity_ids = community_dict.get("entity_ids", [])
            relation_id_set = set(community_dict.get("relation_ids", []))

            entities = await self.get_entities(entity_ids)
            all_relations = await self.graph_backend.get_all_edges()
            relations = [relation for relation in all_relations if relation and relation.id in relation_id_set]
            entities = [entity for entity in entities if entity]
            relations = [relation for relation in relations if relation]

            communities.append(Community(
                id=community_id,
                level=community_dict["level"],
                cluster_id=community_dict["cluster_id"],
                entities=entities,
                relations=relations,
            ))

        return communities

    async def query_entities(self, query: str, top_k: int = 20) -> List[Entity]:
        """
        Search for entities using semantic similarity.

        :param query: Search query text.
        :param top_k: Number of results to return.
        :return: Matching entities.
        """
        query_vector = await self.embedder.embed_text(query)

        results = await self.entity_vector_db.query(Embedding(vector=query_vector), top_k=top_k)
        entity_ids = [r.id for r in results]
        entities = await self.get_entities(entity_ids)
        return [e for e in entities if e is not None]

    async def query_relations(self, query: str, top_k: int = 20) -> List[Relation]:
        """
        Search for relations using semantic similarity.

        :param query: Search query text.
        :param top_k: Number of results to return.
        :return: Matching relations.
        """
        query_vector = await self.embedder.embed_text(query)

        results = await self.relation_vector_db.query(Embedding(vector=query_vector), top_k=top_k)
        edge_specs: List[EdgeSpec] = [
            (
                str(r.metadata.get("subject")),
                str(r.metadata.get("object")),
                r.id,
            )
            for r in results
            if r.metadata.get("subject") and r.metadata.get("object")
        ]
        if not edge_specs:
            return []
        relations = await self.get_relations(edge_specs)
        return [r for r in relations if r is not None]

    async def _build_embeddings(
            self, 
            payloads: Dict[str, Dict[str, Any]], 
            tqdm_title: Optional[str]="Vectorization"
    ) -> List[Embedding]:
        """
        Convert text payloads to vector embeddings.

        :param payloads: Mapping ``id -> payload`` where payload contains ``content``.
        :return: Embeddings ready for vector storage upsert.
        """
        if not self.embedder:
            raise ValueError("Embedder is required for vector operations.")
        if not payloads:
            return []

        payload_items = list(payloads.items())
        contents: List[str] = [str(payload.get("content", "")) for _, payload in payload_items]
        vectors = await self.embedder.batch_embed_text(contents, desc=tqdm_title)

        embeddings: List[Embedding] = []
        for (record_id, payload), vector in zip(payload_items, vectors):
            metadata = {key: value for key, value in payload.items() if key != "content"}
            embeddings.append(
                Embedding(
                    id=record_id,
                    vector=vector,
                    metadata=metadata,
                )
            )
        return embeddings

    async def _validate_relation_endpoints_exist(self, relations: List[Relation]) -> None:
        """
        Validate that all relation endpoints exist as entities.

        :param relations: Relations whose subject/object IDs must exist as nodes.
        :raises ValueError: If at least one referenced entity is missing.
        """
        all_entity_ids: set[str] = set()
        for relation in relations:
            all_entity_ids.add(relation.subject_id)
            all_entity_ids.add(relation.object_id)

        existing_entities = await self.graph_backend.get_nodes(list(all_entity_ids))
        existing_ids = {entity.id for entity in existing_entities if entity is not None}
        missing_ids = all_entity_ids - existing_ids

        if missing_ids:
            raise ValueError(
                f"Cannot insert/update relations referencing non-existent entities: {missing_ids}"
            )

    async def _get_existing_relations_grouped_by_id(
        self,
        relation_ids: Set[str],
    ) -> Dict[str, List[Relation]]:
        """
        Retrieve existing relations grouped by relation ID.

        :param relation_ids: Relation IDs to search in graph storage.
        :return: Mapping from relation ID to all matching stored relations.
        """
        if not relation_ids:
            return {}

        all_relations = await self.graph_backend.get_all_edges()
        grouped: Dict[str, List[Relation]] = defaultdict(list)
        for relation in all_relations:
            assert relation  # this should not None due to input args typing
            assert relation.id
            if relation.id in relation_ids:
                grouped[relation.id].append(relation)
        return dict(grouped)

    @staticmethod
    def _unique_description_fragments(descriptions: Iterable[str]) -> List[str]:
        """
        Split descriptions into normalized fragments and keep first-seen unique ones.

        This prevents repeated sentence fragments when previously merged descriptions
        are merged again with incremental upserts.

        :param descriptions: Description texts to split and normalize.
        :return: Deduplicated description fragments in first-seen order.
        """
        unique_parts: List[str] = []
        seen: set[str] = set()

        for description in descriptions:
            text = (description or "").strip()
            if not text:
                continue
            raw_parts = re.split(r"\n+|(?<=[.!?])\s+", text)
            for part in raw_parts:
                cleaned = re.sub(r"\s+", " ", part).strip()
                if not cleaned:
                    continue
                key = cleaned.casefold()
                if key in seen:
                    continue
                seen.add(key)
                unique_parts.append(cleaned)

        return unique_parts

    @staticmethod
    def _merge_entities(entity_groups: Dict[str, List[Entity]]) -> List[Entity]:
        """
        Merge entities sharing the same ID by combining descriptions and metadata.

        :param entity_groups: Mapping of entity ID to list of entities.
        :return: One merged entity per ID.
        """
        merged: list[Entity] = []

        for entities in entity_groups.values():
            if len(entities) == 1:
                merged.append(entities[0])
                continue

            by_richness = sorted(entities, key=lambda e: len(e.source_chunk_id), reverse=True)
            primary = by_richness[0]

            descriptions = Index._unique_description_fragments(
                [e.description for e in by_richness]
            )

            all_chunks = set[str]()
            all_docs = set[str]()
            all_clusters: list[ClusterInfo] = []
            for e in by_richness:
                all_chunks.update(e.source_chunk_id)
                all_docs.update(e.documents_id)
                all_clusters.extend(e.clusters)

            deduplicated_clusters: List[ClusterInfo] = []
            seen_cluster_keys: Set[tuple[int, int]] = set()
            for cluster in all_clusters:
                assert isinstance(cluster, dict)  # or what is cluster?
                try:
                    level = cluster['level']
                    cluster_id = cluster['cluster_id']
                except (TypeError, ValueError):
                    continue

                cluster_key = (level, cluster_id)
                if cluster_key in seen_cluster_keys:
                    continue

                seen_cluster_keys.add(cluster_key)
                normalized_cluster: ClusterInfo = {"level": level, "cluster_id": cluster_id}
                for key, value in cluster.items():
                    if key not in normalized_cluster:
                        normalized_cluster[key] = value
                deduplicated_clusters.append(normalized_cluster)

            merged.append(Entity(
                id=primary.id,
                entity_name=primary.entity_name,
                entity_type=primary.entity_type,
                description=" ".join(descriptions),
                source_chunk_id=sorted(all_chunks),
                documents_id=sorted(all_docs),
                clusters=deduplicated_clusters,
            ))

        return merged

    @staticmethod
    def _merge_relations(relation_groups: Dict[str, List[Relation]]) -> List[Relation]:
        """
        Merge duplicate relations by combining descriptions and averaging strength.

        For each group, the relation with the most source chunks becomes the primary.
        Unique descriptions are concatenated; relation_strength is averaged;
        source_chunk_ids are unioned.

        :param relation_groups: Mapping of relation ID to list of duplicates.
        :return: One merged relation per group.
        """
        merged: list[Relation] = []

        for relations in relation_groups.values():
            if len(relations) == 1:
                merged.append(relations[0])
                continue

            by_richness = sorted(relations, key=lambda r: len(r.source_chunk_id), reverse=True)
            primary = by_richness[0]

            descriptions = Index._unique_description_fragments(
                [r.description for r in by_richness]
            )

            avg_strength = sum(r.relation_strength for r in by_richness) / len(by_richness)

            all_chunks = set[str]()
            for r in by_richness:
                all_chunks.update(r.source_chunk_id)

            merged.append(Relation(
                id=primary.id,
                subject_id=primary.subject_id,
                object_id=primary.object_id,
                subject_name=primary.subject_name,
                object_name=primary.object_name,
                relation_type=primary.relation_type,
                description=" ".join(descriptions),
                relation_strength=avg_strength,
                source_chunk_id=sorted(all_chunks),
            ))

        return merged

    async def _update_reverse_indexes(
        self,
        entities: Optional[List[Entity]] = None,
        relations: Optional[List[Relation]] = None,
        deleted_entity_ids: Optional[List[str]] = None,
        deleted_relation_ids: Optional[List[str]] = None,
        deleted_chunk_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Incrementally update reverse indexes from changed entities/relations.

        :param entities: Upserted entities to add to chunk-to-entity index.
        :param relations: Upserted relations to add to chunk-to-relation index.
        :param deleted_entity_ids: Entity IDs removed from the graph.
        :param deleted_relation_ids: Relation IDs removed from the graph.
        :param deleted_chunk_ids: Chunk IDs removed from KV/vector storage.
        """
        if deleted_chunk_ids:
            for chunk_id in deleted_chunk_ids:
                self._chunk_to_entities.pop(chunk_id, None)
                self._chunk_to_relations.pop(chunk_id, None)

        if deleted_entity_ids:
            removed_entities = set(deleted_entity_ids)
            for chunk_id, entity_ids in list(self._chunk_to_entities.items()):
                entity_ids.difference_update(removed_entities)
                if not entity_ids:
                    self._chunk_to_entities.pop(chunk_id, None)

        if deleted_relation_ids:
            removed_relations = set(deleted_relation_ids)
            for chunk_id, relation_ids in list(self._chunk_to_relations.items()):
                relation_ids.difference_update(removed_relations)
                if not relation_ids:
                    self._chunk_to_relations.pop(chunk_id, None)

        if entities:
            entities_map = self._get_items_map(entities)
            for chunk_id, entity_ids in entities_map.items():
                self._chunk_to_entities[chunk_id].update(entity_ids)

        if relations:
            relations_map = self._get_items_map(relations)
            for chunk_id, relation_ids in relations_map.items():
                self._chunk_to_relations[chunk_id].update(relation_ids)

    async def _rebuild_reverse_indexes(self) -> None:
        """
        Rebuild reverse indexes by scanning graph data once.

        Used as fallback for cold-start consistency with preloaded graphs.
        """
        self._chunk_to_entities.clear()
        self._chunk_to_relations.clear()

        all_entities = await self.graph_backend.get_all_nodes()
        all_relations = await self.graph_backend.get_all_edges()

        await self._update_reverse_indexes(
            entities=all_entities,
            relations=all_relations,
        )

    async def _find_entities_by_chunk_ids(self, chunk_ids: List[str]) -> List[Entity]:
        """
        Find all entities referencing any of the given chunk IDs.

        :param chunk_ids: Chunk identifiers.
        :return: Entities referencing these chunks.
        """
        if not self._chunk_to_entities:
            await self._rebuild_reverse_indexes()

        entity_ids = set[str]()
        for chunk_id in chunk_ids:
            entity_ids.update(self._chunk_to_entities.get(chunk_id, set()))

        entities = await self.graph_backend.get_nodes(list(entity_ids))
        return [e for e in entities if e is not None]

    @staticmethod
    def _get_items_map(items: List[Entity] | List[Relation]) -> Dict[str, List[str]]:
        """
        Build reverse mapping chunk_id -> list of item IDs using source_chunk_id.

        :param items: Entities or relations with ``id`` and ``source_chunk_id`` fields.
        :return: Mapping from chunk ID to list of entity/relation IDs.
        """
        chunks_map: Dict[str, List[str]] = defaultdict(list)
        for item in items:
            if not item or not item.id:
                continue
            for chunk_id in getattr(item, "source_chunk_id", []):
                chunks_map[chunk_id].append(item.id)
        return dict(chunks_map)

    @staticmethod
    def _unique_relation_ids_from_grouped(relations_by_node: List[List[Relation]]) -> List[str]:
        """
        Flatten grouped relations and return unique relation IDs (first-seen order).

        :param relations_by_node: Relations grouped by source node.
        :return: Unique relation IDs preserving first-seen order.
        """
        relation_ids: List[str] = []
        seen: Set[str] = set()
        for relations in relations_by_node:
            for relation in relations:
                relation_id = getattr(relation, "id", None)
                if not relation_id or relation_id in seen:
                    continue
                seen.add(relation_id)
                relation_ids.append(relation_id)
        return relation_ids

    @staticmethod
    def _build_storage_kwargs(
            storage_folder: str,
            filename: str,
            provided_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build effective storage kwargs and ensure a default absolute ``filename``.

        :param storage_folder: Base folder for storage files.
        :param filename: Default storage filename.
        :param provided_kwargs: Optional custom kwargs from user configuration.
        :return: Final kwargs dictionary for storage backend initialization.
        """
        kwargs = dict(provided_kwargs or {})
        kwargs.setdefault(
            "filename",
            os.path.abspath(os.path.join(storage_folder, filename)),
        )
        return kwargs
