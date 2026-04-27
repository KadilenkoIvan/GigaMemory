from collections import defaultdict
from typing import Any, Dict, List, Optional
import logging

from .conflict_client import TripletConflictClient
from .graph_backend import GraphEdge
from .models import DialogueMemoryState, FactRecord, MemoryFact
from .slot_select_client import SlotSelectClient
from .triplet_client import TripletExtractionClient

logger = logging.getLogger(__name__)


class DSTManager:
    """
    Slot state manager.
    Slots correspond to subgraphs in a single user knowledge graph.
    Facts are stored as KG triplets per slot.

    Conflict resolution strategy (hybrid):
      Rule layer — exact (subject, relation) match → auto-deactivate old record.
      LLM layer  — same subject, ambiguous relation → TripletConflictClient.

    When a ``RaguGraphProcessor`` is provided, every insert/delete is mirrored
    to RAGU's KnowledgeGraph in addition to the in-memory ``DialogueMemoryState``.
    The in-memory state is always the source of truth for conflict resolution and
    slot-based retrieval; RAGU is used for semantic (vector) retrieval.
    """

    def __init__(
        self,
        triplet_extractor: TripletExtractionClient,
        slot_selector: SlotSelectClient,
        *,
        conflict_resolver: Optional[TripletConflictClient] = None,
        single_pass_fallback: bool = True,
        ragu_processor: Optional[Any] = None,
    ):
        self._states: Dict[str, DialogueMemoryState] = {}
        self.triplet_extractor = triplet_extractor
        self.slot_selector = slot_selector
        self.conflict_resolver = conflict_resolver
        self.single_pass_fallback = single_pass_fallback
        # Optional RAGU backend (RaguGraphProcessor); imported lazily to avoid
        # hard dependency when RAGU is not configured.
        self.ragu_processor = ragu_processor

    def get_state(self, dialogue_id: str) -> DialogueMemoryState:
        if dialogue_id not in self._states:
            logger.info("Creating new dialogue state dialogue_id=%s", dialogue_id)
            self._states[dialogue_id] = DialogueMemoryState(dialogue_id=dialogue_id)
        return self._states[dialogue_id]

    def upsert_from_message(self, dialogue_id: str, user_text: str) -> List[MemoryFact]:
        state = self.get_state(dialogue_id)
        state.step += 1
        logger.debug(
            "DST upsert start dialogue_id=%s step=%d text_len=%d",
            dialogue_id,
            state.step,
            len(user_text),
        )
        created: List[MemoryFact] = []
        selected_slots = self.slot_selector.select_slots(user_text)
        triplets = []
        for slot in selected_slots:
            triplets.extend(self.triplet_extractor.extract_for_slot(user_text, slot))
        if not triplets and self.single_pass_fallback:
            triplets = self.triplet_extractor.extract(user_text)
        if not triplets:
            logger.info("No triplets extracted dialogue_id=%s step=%d", dialogue_id, state.step)
            return [], selected_slots

        # Group new triplets by slot for per-slot conflict resolution
        by_slot: Dict[str, List] = defaultdict(list)
        for t in triplets:
            by_slot[t.slot].append(t)

        for slot, slot_triplets in by_slot.items():
            if slot not in state.slots:
                state.slots[slot] = []

            # --- Conflict resolution ---
            skip_indices: set = set()
            deactivated_record_ids: List[int] = []

            if self.conflict_resolver is not None:
                # Build GraphEdge list directly from DialogueMemoryState —
                # no separate graph backend needed for conflict resolution.
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
                    for rec in state.slots.get(slot, [])
                    if rec.is_active and rec.subject in subjects
                ]

                if all_existing:
                    resolution = self.conflict_resolver.resolve(
                        slot, all_existing, slot_triplets
                    )
                    # Apply deactivations to state records
                    for rid in resolution.deactivate_ids:
                        for rec in state.slots.get(slot, []):
                            if rec.record_id == rid and rec.is_active:
                                rec.is_active = False
                                rec.updated_at_step = state.step
                                deactivated_record_ids.append(rid)
                                logger.info(
                                    "Deactivated record_id=%d slot=%s step=%d",
                                    rid, slot, state.step,
                                )
                    skip_indices = resolution.skip_new_indices

            # Mirror deactivations to RAGU
            if self.ragu_processor is not None and deactivated_record_ids:
                from .ragu_graph_processor import GraphTripletDelete
                self.ragu_processor.delete_triplet_deltas([
                    GraphTripletDelete(
                        record_id=rid,
                        dialogue_id=dialogue_id,
                        slot=slot,
                    )
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
                    ))

                created.append(
                    MemoryFact(
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
                    )
                )

            # Mirror inserts to RAGU in one batch
            if self.ragu_processor is not None and new_deltas:
                self.ragu_processor.upsert_triplet_deltas(new_deltas)

        return created, selected_slots

    def deactivate_record(self, dialogue_id: str, record_id: int) -> bool:
        """
        Deactivate a single fact record (soft-delete).
        Returns True if the record was found and deactivated.
        """
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
                GraphTripletDelete(
                    record_id=record_id,
                    dialogue_id=dialogue_id,
                    slot=found_slot,
                )
            ])
        return True

    def active_slot_names(self, dialogue_id: str) -> List[str]:
        """Имена слотов, в которых есть хотя бы одна активная запись (стабильный порядок)."""
        state = self.get_state(dialogue_id)
        names: List[str] = []
        for slot, records in state.slots.items():
            if any(r.is_active for r in records):
                names.append(slot)
        return sorted(names)

    def memory_lines_for_slots(
        self, dialogue_id: str, slot_names: List[str]
    ) -> List[str]:
        """
        Строки «слот: значение» для финальной LLM по выбранным слотам
        (все активные записи по порядку в state).
        """
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
        result: List[MemoryFact] = []
        for slot, records in state.slots.items():
            for rec in records:
                if rec.is_active:
                    result.append(
                        MemoryFact(
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
                        )
                    )
        return result

    def entity_scope_for_slots(
        self, dialogue_id: str, slot_names: List[str], hops: int = 1
    ) -> List[str]:
        """
        Returns entity names active in the given slots.

        With RAGU enabled, LocalSearchEngine handles graph expansion internally,
        so this returns a flat list of entity names for informational purposes only.
        Without RAGU, returns an empty list (legacy vector store handled entity scope).
        """
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
        Memory as an ordered list of slots; each slot holds a list of saved
        user messages (full message text per FactRecord, in chronological
        append order for that slot).

        [
          {"slot": "имя слота", "messages": [ {...}, ... ]},
          ...
        ]
        """
        state = self.get_state(dialogue_id)
        result: List[Dict[str, Any]] = []
        for slot, records in state.slots.items():
            messages: List[dict] = []
            for rec in records:
                if not rec.is_active:
                    continue
                messages.append(
                    {
                        "record_id": rec.record_id,
                        "message_text": rec.value,
                        "source_text": rec.source_text,
                        "subject": rec.subject,
                        "relation": rec.relation,
                        "object": rec.object,
                        "created_at_step": rec.created_at_step,
                        "updated_at_step": rec.updated_at_step,
                        "is_active": rec.is_active,
                    }
                )
            result.append({"slot": slot, "messages": messages})
        return result

    def clear_dialogue(self, dialogue_id: str) -> None:
        logger.info("Clearing DST state dialogue_id=%s", dialogue_id)
        self._states.pop(dialogue_id, None)
