from collections import defaultdict
from typing import Any, Dict, List, Optional
import logging

from .conflict_client import TripletConflictClient
from .graph_backend import UserGraphBackend
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
    """

    def __init__(
        self,
        triplet_extractor: TripletExtractionClient,
        slot_selector: SlotSelectClient,
        *,
        conflict_resolver: Optional[TripletConflictClient] = None,
        single_pass_fallback: bool = True,
    ):
        self._states: Dict[str, DialogueMemoryState] = {}
        self.triplet_extractor = triplet_extractor
        self.slot_selector = slot_selector
        self.conflict_resolver = conflict_resolver
        self.single_pass_fallback = single_pass_fallback
        self.graph = UserGraphBackend()

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
            return []

        # Group new triplets by slot for per-slot conflict resolution
        by_slot: Dict[str, List] = defaultdict(list)
        for t in triplets:
            by_slot[t.slot].append(t)

        for slot, slot_triplets in by_slot.items():
            if slot not in state.slots:
                state.slots[slot] = []

            # --- Conflict resolution ---
            skip_indices: set = set()
            if self.conflict_resolver is not None:
                # Gather all existing active edges for subjects present in new triplets
                subjects = {t.subject for t in slot_triplets}
                all_existing = []
                for subj in subjects:
                    all_existing.extend(
                        self.graph.get_active_by_subject_in_slot(dialogue_id, slot, subj)
                    )

                if all_existing:
                    resolution = self.conflict_resolver.resolve(
                        slot, all_existing, slot_triplets
                    )
                    # Apply deactivations to graph and state records
                    for rid in resolution.deactivate_ids:
                        self.graph.deactivate_record(dialogue_id, rid)
                        for rec in state.slots.get(slot, []):
                            if rec.record_id == rid:
                                rec.is_active = False
                                rec.updated_at_step = state.step
                                logger.info(
                                    "Deactivated record_id=%d slot=%s step=%d",
                                    rid, slot, state.step,
                                )
                    skip_indices = resolution.skip_new_indices

            # --- Insert surviving new triplets ---
            for idx, t in enumerate(slot_triplets):
                if idx in skip_indices:
                    logger.debug("Skipping duplicate triplet idx=%d (%s|%s|%s)",
                                 idx, t.subject, t.relation, t.object)
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
                self.graph.upsert_triplet(
                    dialogue_id,
                    record_id=rid,
                    slot=slot,
                    subject=t.subject,
                    relation=t.relation,
                    object_=t.object,
                )
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
        return created

    def deactivate_record(self, dialogue_id: str, record_id: int) -> bool:
        """
        Deactivate a single fact record (soft-delete).
        Used by conflict_resolver and can also be called externally.
        Returns True if the record was found and deactivated.
        """
        state = self.get_state(dialogue_id)
        graph_ok = self.graph.deactivate_record(dialogue_id, record_id)
        for records in state.slots.values():
            for rec in records:
                if rec.record_id == record_id:
                    rec.is_active = False
                    return True
        return graph_ok

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
        seeds = self.graph.entities_for_slots(dialogue_id, slot_names)
        return self.graph.expand_entities(dialogue_id, seeds, hops=hops)

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
        self.graph.clear_dialogue(dialogue_id)
