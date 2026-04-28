from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import logging

from .conflict_client import TripletConflictClient
from .config import SLOT_DEFAULT_TTL
from .graph_backend import GraphEdge
from .models import DialogueMemoryState, FactRecord, MemoryFact, is_expired, now_iso
from .slot_select_client import SlotSelectClient
from .triplet_client import ExtractedTriplet, TripletExtractionClient

logger = logging.getLogger(__name__)


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not v1 or not v2:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def _triplet_text(subject: str, relation: str, obj: str) -> str:
    return f"{subject} {relation} {obj}"


class DSTManager:
    """
    Slot state manager.
    Slots correspond to subgraphs in a single user knowledge graph.
    Facts are stored as KG triplets per slot.

    Conflict resolution strategy (hybrid):
      Rule layer — exact (subject, relation) match → auto-deactivate old record.
      LLM layer  — same subject, ambiguous relation → TripletConflictClient.
      Semantic layer — cosine similarity > threshold → deactivate near-duplicate, insert new.

    TTL expiry is checked lazily on each read operation (soft-delete expired records).

    When a ``RaguGraphProcessor`` is provided, every insert/delete is mirrored
    to RAGU's KnowledgeGraph in addition to the in-memory ``DialogueMemoryState``.
    """

    def __init__(
        self,
        triplet_extractor: TripletExtractionClient,
        slot_selector: SlotSelectClient,
        *,
        conflict_resolver: Optional[TripletConflictClient] = None,
        single_pass_fallback: bool = True,
        ragu_processor: Optional[Any] = None,
        ttl_mode: str = "mode2",
        ttl_slot_overrides: Optional[Dict[str, str]] = None,
        semantic_dedup_enabled: bool = True,
        semantic_dedup_threshold: float = 0.9,
    ):
        self._states: Dict[str, DialogueMemoryState] = {}
        self.triplet_extractor = triplet_extractor
        self.slot_selector = slot_selector
        self.conflict_resolver = conflict_resolver
        self.single_pass_fallback = single_pass_fallback
        self.ragu_processor = ragu_processor
        self.ttl_mode = ttl_mode
        self.semantic_dedup_enabled = semantic_dedup_enabled
        self.semantic_dedup_threshold = semantic_dedup_threshold

        # Build effective TTL map: defaults + overrides
        self._slot_ttl: Dict[str, str] = dict(SLOT_DEFAULT_TTL)
        if ttl_slot_overrides:
            self._slot_ttl.update(ttl_slot_overrides)

    def _resolve_ttl(self, triplet: ExtractedTriplet) -> str:
        """Determine TTL for a triplet based on ttl_mode."""
        if self.ttl_mode == "mode2" and triplet.ttl and triplet.ttl != "inf":
            return triplet.ttl
        if self.ttl_mode == "mode2" and triplet.ttl == "inf":
            # Model explicitly said inf — respect it
            return "inf"
        # mode1 or fallback: use slot default
        return self._slot_ttl.get(triplet.slot, "inf")

    def _expire_slot(self, state: DialogueMemoryState, slot: str) -> List[int]:
        """
        Soft-delete expired records in a slot.
        Returns list of deactivated record_ids (for RAGU sync).
        """
        deactivated: List[int] = []
        for rec in state.slots.get(slot, []):
            if rec.is_active and rec.is_expired():
                rec.is_active = False
                rec.updated_at_step = state.step
                deactivated.append(rec.record_id)
                logger.info(
                    "TTL expired: deactivated record_id=%d slot=%s ttl=%s created=%s",
                    rec.record_id, slot, rec.ttl, rec.created_at_datetime,
                )
        return deactivated

    def _expire_all(self, state: DialogueMemoryState) -> Dict[str, List[int]]:
        """Expire all slots. Returns {slot: [deactivated_ids]}."""
        result: Dict[str, List[int]] = {}
        for slot in list(state.slots.keys()):
            deactivated = self._expire_slot(state, slot)
            if deactivated:
                result[slot] = deactivated
        return result

    def _sync_expirations_to_ragu(self, dialogue_id: str, expired_by_slot: Dict[str, List[int]]) -> None:
        if self.ragu_processor is None:
            return
        from .ragu_graph_processor import GraphTripletDelete
        all_deletes = []
        for slot, rids in expired_by_slot.items():
            for rid in rids:
                all_deletes.append(GraphTripletDelete(
                    record_id=rid, dialogue_id=dialogue_id, slot=slot
                ))
        if all_deletes:
            self.ragu_processor.delete_triplet_deltas(all_deletes)

    def _semantic_dedup(
        self,
        new_triplet: ExtractedTriplet,
        existing_active: List[FactRecord],
    ) -> Optional[FactRecord]:
        """
        Find an existing active record semantically similar to new_triplet.
        Returns the closest record if cosine similarity >= threshold, else None.
        Only checks within the same slot (caller ensures this).
        """
        if not self.semantic_dedup_enabled:
            return None
        if self.ragu_processor is None or not hasattr(self.ragu_processor, "embed_text_sync"):
            return None
        if not existing_active:
            return None

        new_text = _triplet_text(new_triplet.subject, new_triplet.relation, new_triplet.object)
        new_emb = self.ragu_processor.embed_text_sync(new_text)
        if not new_emb:
            return None

        best_sim = 0.0
        best_rec: Optional[FactRecord] = None
        for rec in existing_active:
            rec_text = _triplet_text(rec.subject, rec.relation, rec.object)
            rec_emb = self.ragu_processor.embed_text_sync(rec_text)
            if not rec_emb:
                continue
            sim = _cosine_similarity(new_emb, rec_emb)
            if sim > best_sim:
                best_sim = sim
                best_rec = rec

        if best_sim >= self.semantic_dedup_threshold and best_rec is not None:
            logger.info(
                "Semantic dedup: sim=%.4f >= %.2f | new=[%s|%s|%s] vs existing record_id=%d [%s|%s|%s]",
                best_sim, self.semantic_dedup_threshold,
                new_triplet.subject, new_triplet.relation, new_triplet.object,
                best_rec.record_id,
                best_rec.subject, best_rec.relation, best_rec.object,
            )
            return best_rec
        return None

    def get_state(self, dialogue_id: str) -> DialogueMemoryState:
        if dialogue_id not in self._states:
            logger.info("Creating new dialogue state dialogue_id=%s", dialogue_id)
            self._states[dialogue_id] = DialogueMemoryState(dialogue_id=dialogue_id)
        return self._states[dialogue_id]

    def upsert_from_message(
        self, dialogue_id: str, user_text: str
    ) -> Tuple[List[MemoryFact], List[str]]:
        state = self.get_state(dialogue_id)
        state.step += 1
        logger.debug(
            "DST upsert start dialogue_id=%s step=%d text_len=%d",
            dialogue_id, state.step, len(user_text),
        )

        # Expire stale records on write too (belt-and-suspenders)
        expired = self._expire_all(state)
        if expired:
            self._sync_expirations_to_ragu(dialogue_id, expired)

        created: List[MemoryFact] = []
        selected_slots = self.slot_selector.select_slots(user_text)
        triplets: List[ExtractedTriplet] = []
        for slot in selected_slots:
            triplets.extend(self.triplet_extractor.extract_for_slot(user_text, slot))
        if not triplets and self.single_pass_fallback:
            triplets = self.triplet_extractor.extract(user_text)
        if not triplets:
            logger.info("No triplets extracted dialogue_id=%s step=%d", dialogue_id, state.step)
            return [], selected_slots

        by_slot: Dict[str, List[ExtractedTriplet]] = defaultdict(list)
        for t in triplets:
            by_slot[t.slot].append(t)

        for slot, slot_triplets in by_slot.items():
            if slot not in state.slots:
                state.slots[slot] = []

            # Expire slot before processing (lazy expiry)
            expired_in_slot = self._expire_slot(state, slot)
            if expired_in_slot and self.ragu_processor is not None:
                from .ragu_graph_processor import GraphTripletDelete
                self.ragu_processor.delete_triplet_deltas([
                    GraphTripletDelete(record_id=rid, dialogue_id=dialogue_id, slot=slot)
                    for rid in expired_in_slot
                ])

            # --- Semantic dedup PRE-PASS ---
            # For each new triplet: check if a very similar active record exists.
            # If yes: deactivate old, insert new (refreshed TTL timer).
            semantic_dedup_deactivate: List[int] = []
            semantic_dedup_skip_new: set = set()  # do NOT skip — we insert new (replaces old)

            active_in_slot = [r for r in state.slots[slot] if r.is_active]

            if self.semantic_dedup_enabled:
                for idx, new_t in enumerate(slot_triplets):
                    similar_rec = self._semantic_dedup(new_t, active_in_slot)
                    if similar_rec is not None and similar_rec.record_id not in semantic_dedup_deactivate:
                        semantic_dedup_deactivate.append(similar_rec.record_id)
                        logger.info(
                            "Semantic dedup: marking record_id=%d for deactivation (new triplet refreshes it)",
                            similar_rec.record_id,
                        )

                # Apply semantic dedup deactivations
                for rid in semantic_dedup_deactivate:
                    for rec in state.slots[slot]:
                        if rec.record_id == rid and rec.is_active:
                            rec.is_active = False
                            rec.updated_at_step = state.step

                if semantic_dedup_deactivate and self.ragu_processor is not None:
                    from .ragu_graph_processor import GraphTripletDelete
                    self.ragu_processor.delete_triplet_deltas([
                        GraphTripletDelete(record_id=rid, dialogue_id=dialogue_id, slot=slot)
                        for rid in semantic_dedup_deactivate
                    ])

            # Rebuild active list after semantic dedup
            active_in_slot = [r for r in state.slots[slot] if r.is_active]

            # --- Conflict resolution ---
            skip_indices: set = set()
            deactivated_record_ids: List[int] = []

            if self.conflict_resolver is not None:
                subjects = {t.subject for t in slot_triplets}
                all_existing: List[GraphEdge] = [
                    GraphEdge(
                        edge_id=rec.record_id,
                        slot=slot,
                        subject=rec.subject,
                        relation=rec.relation,
                        object=rec.object,
                        record_id=rec.record_id,
                        is_active=rec.is_active,
                    )
                    for rec in active_in_slot
                    if rec.subject in subjects
                ]

                if all_existing:
                    resolution = self.conflict_resolver.resolve(slot, all_existing, slot_triplets)
                    for rid in resolution.deactivate_ids:
                        for rec in state.slots.get(slot, []):
                            if rec.record_id == rid and rec.is_active:
                                rec.is_active = False
                                rec.updated_at_step = state.step
                                deactivated_record_ids.append(rid)
                                logger.info("Deactivated record_id=%d slot=%s step=%d", rid, slot, state.step)
                    skip_indices = resolution.skip_new_indices

            if self.ragu_processor is not None and deactivated_record_ids:
                from .ragu_graph_processor import GraphTripletDelete
                self.ragu_processor.delete_triplet_deltas([
                    GraphTripletDelete(record_id=rid, dialogue_id=dialogue_id, slot=slot)
                    for rid in deactivated_record_ids
                ])

            # --- Insert surviving new triplets ---
            new_deltas = []
            for idx, t in enumerate(slot_triplets):
                if idx in skip_indices:
                    logger.debug(
                        "Skipping duplicate triplet idx=%d (%s|%s|%s)",
                        idx, t.subject, t.relation, t.object,
                    )
                    continue

                ttl = self._resolve_ttl(t)
                rid = state.next_record_id
                state.next_record_id += 1
                value = t.as_line()
                rec = FactRecord(
                    record_id=rid,
                    value=value,
                    source_text=user_text,
                    created_at_step=state.step,
                    updated_at_step=state.step,
                    subject=t.subject,
                    relation=t.relation,
                    object=t.object,
                    is_active=True,
                    ttl=ttl,
                    created_at_datetime=now_iso(),
                )
                state.slots[slot].append(rec)

                if self.ragu_processor is not None:
                    from .ragu_graph_processor import GraphTripletDelta
                    new_deltas.append(GraphTripletDelta(
                        record_id=rid,
                        dialogue_id=dialogue_id,
                        step=state.step,
                        slot=slot,
                        subject=t.subject,
                        relation=t.relation,
                        object=t.object,
                        ttl=ttl,
                    ))

                created.append(MemoryFact(
                    slot=slot,
                    record_id=rid,
                    value=value,
                    source_text=user_text,
                    created_at_step=state.step,
                    updated_at_step=state.step,
                    is_active=True,
                    subject=t.subject,
                    relation=t.relation,
                    object=t.object,
                    ttl=ttl,
                    created_at_datetime=rec.created_at_datetime,
                ))

            if self.ragu_processor is not None and new_deltas:
                self.ragu_processor.upsert_triplet_deltas(new_deltas)

        return created, selected_slots

    def deactivate_record(self, dialogue_id: str, record_id: int) -> bool:
        state = self.get_state(dialogue_id)
        found_slot: Optional[str] = None
        for slot, records in state.slots.items():
            for rec in records:
                if rec.record_id == record_id and rec.is_active:
                    rec.is_active = False
                    found_slot = slot
                    break
            if found_slot is not None:
                break

        if found_slot is None:
            return False

        if self.ragu_processor is not None:
            from .ragu_graph_processor import GraphTripletDelete
            self.ragu_processor.delete_triplet_deltas([
                GraphTripletDelete(record_id=record_id, dialogue_id=dialogue_id, slot=found_slot)
            ])
        return True

    def active_slot_names(self, dialogue_id: str) -> List[str]:
        state = self.get_state(dialogue_id)
        # Lazy expiry on read
        expired = self._expire_all(state)
        if expired:
            self._sync_expirations_to_ragu(dialogue_id, expired)

        names: List[str] = []
        for slot, records in state.slots.items():
            if any(r.is_active for r in records):
                names.append(slot)
        return sorted(names)

    def memory_lines_for_slots(self, dialogue_id: str, slot_names: List[str]) -> List[str]:
        state = self.get_state(dialogue_id)
        lines: List[str] = []
        for name in slot_names:
            recs = state.slots.get(name)
            if not recs:
                continue
            for rec in recs:
                if rec.is_active:
                    lines.append(f"{name}: {rec.value}")
        return lines

    def active_facts(self, dialogue_id: str) -> List[MemoryFact]:
        state = self.get_state(dialogue_id)
        # Lazy expiry on read
        expired = self._expire_all(state)
        if expired:
            self._sync_expirations_to_ragu(dialogue_id, expired)

        result: List[MemoryFact] = []
        for slot, records in state.slots.items():
            for rec in records:
                if rec.is_active:
                    result.append(MemoryFact(
                        slot=slot,
                        record_id=rec.record_id,
                        value=rec.value,
                        source_text=rec.source_text,
                        created_at_step=rec.created_at_step,
                        updated_at_step=rec.updated_at_step,
                        is_active=rec.is_active,
                        subject=rec.subject,
                        relation=rec.relation,
                        object=rec.object,
                        ttl=rec.ttl,
                        created_at_datetime=rec.created_at_datetime,
                    ))
        return result

    def expired_facts(self, dialogue_id: str) -> List[Dict[str, Any]]:
        """Return all expired (soft-deleted) records for logging/visualization."""
        state = self.get_state(dialogue_id)
        result = []
        for slot, records in state.slots.items():
            for rec in records:
                if not rec.is_active and is_expired(rec.ttl, rec.created_at_datetime):
                    result.append({
                        "slot": slot,
                        "record_id": rec.record_id,
                        "subject": rec.subject,
                        "relation": rec.relation,
                        "object": rec.object,
                        "ttl": rec.ttl,
                        "created_at_datetime": rec.created_at_datetime,
                        "expired": True,
                    })
        return result

    def entity_scope_for_slots(self, dialogue_id: str, slot_names: List[str], hops: int = 1) -> List[str]:
        state = self.get_state(dialogue_id)
        entities: List[str] = []
        seen: set = set()
        for slot in slot_names:
            for rec in state.slots.get(slot, []):
                if not rec.is_active:
                    continue
                for name in (rec.subject, rec.object):
                    if name and name not in seen:
                        seen.add(name)
                        entities.append(name)
        return entities

    def slots_with_messages(self, dialogue_id: str) -> List[Dict[str, Any]]:
        """
        Memory as an ordered list of slots with active records.
        Triggers lazy TTL expiry before building the payload.
        """
        state = self.get_state(dialogue_id)
        # Lazy expiry on read
        expired = self._expire_all(state)
        if expired:
            self._sync_expirations_to_ragu(dialogue_id, expired)

        result: List[Dict[str, Any]] = []
        for slot, records in state.slots.items():
            messages: List[dict] = []
            for rec in records:
                if not rec.is_active:
                    continue
                messages.append({
                    "record_id": rec.record_id,
                    "message_text": rec.value,
                    "source_text": rec.source_text,
                    "subject": rec.subject,
                    "relation": rec.relation,
                    "object": rec.object,
                    "created_at_step": rec.created_at_step,
                    "updated_at_step": rec.updated_at_step,
                    "is_active": rec.is_active,
                    "ttl": rec.ttl,
                    "created_at_datetime": rec.created_at_datetime,
                })
            result.append({"slot": slot, "messages": messages})
        return result

    def clear_dialogue(self, dialogue_id: str) -> None:
        logger.info("Clearing DST state dialogue_id=%s", dialogue_id)
        self._states.pop(dialogue_id, None)
