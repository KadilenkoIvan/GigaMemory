from typing import Any, Dict, List
import logging

from .models import DialogueMemoryState, FactRecord, MemoryFact
from .slot_client import SlotDecisionClient
from .slot_update_client import SlotOperation, SlotUpdateClient

logger = logging.getLogger(__name__)


class DSTManager:
    """
    Slot state manager.
    TODO: replace slot extraction and delete decision with stronger LLM logic.
    """

    def __init__(self, slot_client: SlotDecisionClient, slot_update: SlotUpdateClient):
        self._states: Dict[str, DialogueMemoryState] = {}
        self.slot_client = slot_client
        self.slot_update = slot_update

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
        existing_slots = list(state.slots.keys())
        slot_decisions = self.slot_client.decide_slots(
            existing_slots=existing_slots,
            user_message=user_text,
        )
        if not slot_decisions:
            logger.info("No slot decisions dialogue_id=%s step=%d", dialogue_id, state.step)
            return []

        created: List[MemoryFact] = []
        for decision in slot_decisions:
            slot = decision.slot_name
            if slot not in state.slots:
                state.slots[slot] = []
            existing = [
                {"id": r.record_id, "value": r.value}
                for r in state.slots[slot]
                if r.is_active
            ]
            ops = self.slot_update.plan_operations(slot, existing, user_text)
            self._apply_operations(state, slot, ops, user_text, created)
        return created

    def _apply_operations(
        self,
        state: DialogueMemoryState,
        slot: str,
        ops: List[SlotOperation],
        source_text: str,
        out_facts: List[MemoryFact],
    ) -> None:
        if slot not in state.slots:
            state.slots[slot] = []

        by_id: Dict[int, FactRecord] = {
            r.record_id: r for r in state.slots[slot] if r.is_active
        }

        for op in ops:
            if op.op == "nothing":
                logger.info(
                    "DST nothing dialogue_id=%s slot=%s step=%d",
                    state.dialogue_id,
                    slot,
                    state.step,
                )
                continue
            if op.op == "add" and op.value:
                rid = state.next_record_id
                state.next_record_id += 1
                rec = FactRecord(
                    record_id=rid,
                    value=op.value,
                    source_text=source_text,
                    created_at_step=state.step,
                    updated_at_step=state.step,
                    is_active=True,
                )
                state.slots[slot].append(rec)
                logger.info(
                    "DST add dialogue_id=%s slot=%s id=%d value_preview=%s",
                    state.dialogue_id,
                    slot,
                    rid,
                    op.value[:80],
                )
                out_facts.append(
                    MemoryFact(
                        slot=slot,
                        record_id=rid,
                        value=op.value,
                        source_text=source_text,
                        created_at_step=state.step,
                        updated_at_step=state.step,
                        is_active=True,
                    )
                )
            elif op.op == "update" and op.record_id is not None and op.value:
                rec = by_id.get(op.record_id)
                if not rec:
                    logger.info(
                        "DST update skipped (unknown id) dialogue_id=%s slot=%s id=%s",
                        state.dialogue_id,
                        slot,
                        op.record_id,
                    )
                    continue
                rec.value = op.value
                rec.source_text = source_text
                rec.updated_at_step = state.step
                logger.info(
                    "DST update dialogue_id=%s slot=%s id=%d value_preview=%s",
                    state.dialogue_id,
                    slot,
                    op.record_id,
                    op.value[:80],
                )
                out_facts.append(
                    MemoryFact(
                        slot=slot,
                        record_id=op.record_id,
                        value=op.value,
                        source_text=source_text,
                        created_at_step=rec.created_at_step,
                        updated_at_step=rec.updated_at_step,
                        is_active=True,
                    )
                )
            elif op.op == "delete" and op.record_id is not None:
                rec = by_id.get(op.record_id)
                if not rec:
                    logger.info(
                        "DST delete skipped (unknown id) dialogue_id=%s slot=%s id=%s",
                        state.dialogue_id,
                        slot,
                        op.record_id,
                    )
                    continue
                rec.is_active = False
                rec.source_text = source_text
                rec.updated_at_step = state.step
                logger.info(
                    "DST delete dialogue_id=%s slot=%s id=%d",
                    state.dialogue_id,
                    slot,
                    op.record_id,
                )
            else:
                logger.debug(
                    "DST op ignored dialogue_id=%s slot=%s op=%s",
                    state.dialogue_id,
                    slot,
                    op,
                )

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
                        )
                    )
        return result

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
