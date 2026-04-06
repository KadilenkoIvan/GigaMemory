from typing import Any, Dict, List
import logging

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
    """

    def __init__(
        self,
        triplet_extractor: TripletExtractionClient,
        slot_selector: SlotSelectClient,
        *,
        single_pass_fallback: bool = True,
    ):
        self._states: Dict[str, DialogueMemoryState] = {}
        self.triplet_extractor = triplet_extractor
        self.slot_selector = slot_selector
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

        for t in triplets:
            slot = t.slot
            if slot not in state.slots:
                state.slots[slot] = []
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

    def delete_with_stub_policy(self, dialogue_id: str, user_text: str) -> None:
        """
        TODO: implement LLM-based delete/update policy for contradictory facts.
        Current placeholder intentionally does nothing.
        """
        _ = dialogue_id
        _ = user_text
        logger.debug("DST delete stub invoked dialogue_id=%s", dialogue_id)

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
