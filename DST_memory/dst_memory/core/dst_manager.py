from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

from ..triplets.conflict_client import TripletConflictClient
from .config import SLOT_DEFAULT_TTL
from .graph_backend import GraphEdge
from .models import DialogueMemoryState, FactRecord, MemoryFact, DeletedFact, is_expired, now_iso
from ..slots.slot_select_client import SlotSelectClient
from ..triplets.triplet_client import DeletionSignal, ExtractedTriplet, TripletExtractionClient

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
      Rule layer  — exact (subject, relation) match → auto-deactivate old record.
      LLM layer   — same subject, ambiguous relation → TripletConflictClient.
      Semantic layer — cosine similarity > threshold → deactivate near-duplicate, insert new.

    Deletion modes (triplet_deletion_mode):
      "none"         — no deletion (default, current behaviour).
      "heuristic"    — Variant C: rule-based negation patterns via NegationDeletionDetector.
      "llm_inline"   — Variant A: delete signals embedded in extraction call (requires context).
      "llm_separate" — Variant B: separate LLM call for deletion after extraction.

    When slot_context_enabled=True, the current slot facts are passed to the
    extraction model, which can then produce richer updates (historical facts,
    explicit deletes).

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
        single_pass_fallback: bool = False,  # legacy; kept for back-compat only
        slot_fallback_on_no_slots: bool = True,
        triplet_fallback_on_empty: bool = True,
        ragu_processor: Optional[Any] = None,
        ttl_mode: str = "mode2",
        ttl_slot_overrides: Optional[Dict[str, str]] = None,
        semantic_dedup_enabled: bool = True,
        semantic_dedup_threshold: float = 0.9,
        # --- Deletion / context ---
        slot_context_enabled: bool = False,
        slot_context_max_facts: int = 10,
        triplet_deletion_mode: str = "none",
        negation_detector: Optional[Any] = None,
        deletion_client: Optional[Any] = None,
        force_infinite_ttl: bool = False,
    ):
        self._states: Dict[str, DialogueMemoryState] = {}
        self.triplet_extractor = triplet_extractor
        self.slot_selector = slot_selector
        self.conflict_resolver = conflict_resolver
        # Legacy single_pass_fallback sets both granular flags when True
        self.slot_fallback_on_no_slots = slot_fallback_on_no_slots if not single_pass_fallback else (slot_fallback_on_no_slots or single_pass_fallback)
        self.triplet_fallback_on_empty = triplet_fallback_on_empty if not single_pass_fallback else (triplet_fallback_on_empty or single_pass_fallback)
        self.ragu_processor = ragu_processor
        self.ttl_mode = ttl_mode
        self.semantic_dedup_enabled = semantic_dedup_enabled
        self.semantic_dedup_threshold = semantic_dedup_threshold
        self.slot_context_enabled = slot_context_enabled
        self.slot_context_max_facts = slot_context_max_facts
        self.triplet_deletion_mode = triplet_deletion_mode
        self.negation_detector = negation_detector
        self.deletion_client = deletion_client
        self.force_infinite_ttl = bool(force_infinite_ttl)

        # Enforce: llm_inline requires context
        if triplet_deletion_mode == "llm_inline" and not slot_context_enabled:
            logger.warning(
                "triplet_deletion_mode='llm_inline' requires slot_context_enabled=True; "
                "forcing slot_context_enabled=True"
            )
            self.slot_context_enabled = True

        # Build effective TTL map: defaults + overrides
        self._slot_ttl: Dict[str, str] = dict(SLOT_DEFAULT_TTL)
        if ttl_slot_overrides:
            self._slot_ttl.update(ttl_slot_overrides)

    @staticmethod
    def _as_of_datetime_for_state(state: DialogueMemoryState) -> Optional[datetime]:
        raw = getattr(state, "dataset_clock_iso", None) or None
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    @staticmethod
    def _new_fact_created_at_iso(
        state: DialogueMemoryState,
        override_iso: Optional[str] = None,
    ) -> str:
        if override_iso and str(override_iso).strip():
            return str(override_iso).strip()
        raw = getattr(state, "dataset_clock_iso", None) or None
        if raw:
            return raw
        return now_iso()

    def _resolve_ttl(self, triplet: ExtractedTriplet) -> str:
        """Determine TTL for a triplet based on ttl_mode."""
        if self.force_infinite_ttl:
            return "inf"
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
        Also tracks deleted facts with reason 'ttl_expired'.
        """
        deactivated: List[int] = []
        as_of = self._as_of_datetime_for_state(state)
        for rec in state.slots.get(slot, []):
            if rec.is_active and rec.is_expired(as_of=as_of):
                rec.is_active = False
                rec.updated_at_step = state.step
                deactivated.append(rec.record_id)
                # Track deletion with reason
                deleted_fact = DeletedFact(
                    slot=slot,
                    record_id=rec.record_id,
                    subject=rec.subject,
                    relation=rec.relation,
                    object=rec.object,
                    value=rec.value,
                    source_text=rec.source_text,
                    created_at_step=rec.created_at_step,
                    created_at_datetime=rec.created_at_datetime,
                    deleted_at_step=state.step,
                    deletion_reason='ttl_expired',
                    deletion_source='ttl_checker',
                    deletion_details={'ttl': rec.ttl}
                )
                state.deleted_facts.append(deleted_fact)
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
        from ..storage.ragu_graph_processor import GraphTripletDelete
        all_deletes = []
        for slot, rids in expired_by_slot.items():
            for rid in rids:
                all_deletes.append(GraphTripletDelete(
                    record_id=rid, dialogue_id=dialogue_id, slot=slot
                ))
        if all_deletes:
            self.ragu_processor.delete_triplet_deltas(all_deletes)

    def _apply_deletion_signals(
        self,
        dialogue_id: str,
        state: DialogueMemoryState,
        slot: str,
        signals: List[DeletionSignal],
        deletion_source: str = 'unknown',  # 'llm_inline', 'llm_separate', 'heuristic'
    ) -> List[int]:
        """
        Применить сигналы удаления к активным записям слота.

        Каскадная стратегия:
          1. Точное совпадение: subject + relation + object (после нормализации).
          2. Если точных совпадений нет — совпадение по subject + relation.

        Returns
        -------
        List of deactivated record_ids (для синхронизации RAGU).
        """
        if not signals:
            return []

        deactivated: List[int] = []
        seen: set = set()
        deleted_records_info: List[Tuple[int, str, str, str, str, str]] = []  # record info for tracking

        # Проход 1: точное совпадение
        for sig in signals:
            for rec in state.slots.get(slot, []):
                if rec.record_id in seen or not rec.is_active:
                    continue
                if rec.subject == sig.subject and rec.relation == sig.relation and rec.object == sig.object:
                    rec.is_active = False
                    rec.updated_at_step = state.step
                    deactivated.append(rec.record_id)
                    seen.add(rec.record_id)
                    deleted_records_info.append((
                        rec.record_id, rec.subject, rec.relation, rec.object,
                        rec.value, rec.source_text, rec.created_at_step, rec.created_at_datetime
                    ))
                    logger.info(
                        "DeletionSignal exact match: deactivated record_id=%d slot=%s [%s|%s|%s]",
                        rec.record_id, slot, sig.subject, sig.relation, sig.object,
                    )

        # Проход 2 (каскад): subject + relation если точных совпадений не нашли
        unmatched_signals = [
            sig for sig in signals
            if not any(
                rec.subject == sig.subject and rec.relation == sig.relation and rec.object == sig.object
                and rec.record_id in seen
                for rec in state.slots.get(slot, [])
            )
        ]
        for sig in unmatched_signals:
            for rec in state.slots.get(slot, []):
                if rec.record_id in seen or not rec.is_active:
                    continue
                if rec.subject == sig.subject and rec.relation == sig.relation:
                    rec.is_active = False
                    rec.updated_at_step = state.step
                    deactivated.append(rec.record_id)
                    seen.add(rec.record_id)
                    deleted_records_info.append((
                        rec.record_id, rec.subject, rec.relation, rec.object,
                        rec.value, rec.source_text, rec.created_at_step, rec.created_at_datetime
                    ))
                    logger.info(
                        "DeletionSignal cascade (subj+rel): deactivated record_id=%d slot=%s [%s|%s|%s]",
                        rec.record_id, slot, rec.subject, rec.relation, rec.object,
                    )

        # Track deletions with reason
        for record_info in deleted_records_info:
            rid, subject, relation, obj, value, source_text, created_step, created_dt = record_info
            deleted_fact = DeletedFact(
                slot=slot,
                record_id=rid,
                subject=subject,
                relation=relation,
                object=obj,
                value=value,
                source_text=source_text,
                created_at_step=created_step,
                created_at_datetime=created_dt,
                deleted_at_step=state.step,
                deletion_reason='deletion_signal',
                deletion_source=deletion_source,
                deletion_details={'match_type': 'exact' if any(s.subject == subject and s.relation == relation and s.object == obj for s in signals) else 'cascade'}
            )
            state.deleted_facts.append(deleted_fact)

        if deactivated and self.ragu_processor is not None:
            from ..storage.ragu_graph_processor import GraphTripletDelete
            self.ragu_processor.delete_triplet_deltas([
                GraphTripletDelete(record_id=rid, dialogue_id=dialogue_id, slot=slot)
                for rid in deactivated
            ])

        return deactivated

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
        self,
        dialogue_id: str,
        user_text: str,
        fact_created_at_iso: Optional[str] = None,
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
        no_slots_selected = len(selected_slots) == 0
        triplets: List[ExtractedTriplet] = []
        inline_deletions_by_slot: Dict[str, List[DeletionSignal]] = defaultdict(list)

        for slot in selected_slots:
            if self.slot_context_enabled:
                # Gather current active facts for this slot as context lines
                active_now = [r for r in state.slots.get(slot, []) if r.is_active]
                existing_lines = [
                    r.as_line() for r in active_now[: self.slot_context_max_facts]
                ]
                # enable_deletion=True ONLY for llm_inline mode — other modes
                # (heuristic, llm_separate, none) use context for awareness only,
                # the model should NOT output delete arrays in those modes.
                inline_mode = self.triplet_deletion_mode == "llm_inline"
                slot_triplets, slot_deletions = self.triplet_extractor.extract_for_slot_with_context(
                    user_text, slot, existing_lines, enable_deletion=inline_mode
                )
                if inline_mode:
                    inline_deletions_by_slot[slot].extend(slot_deletions)
            else:
                slot_triplets = self.triplet_extractor.extract_for_slot(user_text, slot)
            triplets.extend(slot_triplets)

        if no_slots_selected:
            # Slot selector returned nothing
            if self.slot_fallback_on_no_slots:
                logger.info(
                    "Slot selector returned no slots dialogue_id=%s step=%d → single-pass fallback",
                    dialogue_id, state.step,
                )
                triplets = self.triplet_extractor.extract(user_text)
            else:
                logger.info(
                    "Slot selector returned no slots dialogue_id=%s step=%d → skipping (fallback disabled)",
                    dialogue_id, state.step,
                )
        elif not triplets:
            # Slots were selected but all per-slot extractions returned empty
            if self.triplet_fallback_on_empty:
                logger.info(
                    "All per-slot extractions returned empty dialogue_id=%s step=%d → single-pass fallback",
                    dialogue_id, state.step,
                )
                triplets = self.triplet_extractor.extract(user_text)
            else:
                logger.info(
                    "All per-slot extractions returned empty dialogue_id=%s step=%d → skipping (fallback disabled)",
                    dialogue_id, state.step,
                )

        if not triplets and not any(inline_deletions_by_slot.values()):
            # Check if any deletion mode might still produce deletions
            has_potential_deletions = (
                self.triplet_deletion_mode in ("heuristic", "llm_separate")
                and any(state.slots.values())
            )
            if not has_potential_deletions:
                logger.info("No triplets extracted dialogue_id=%s step=%d", dialogue_id, state.step)
                return [], selected_slots

        by_slot: Dict[str, List[ExtractedTriplet]] = defaultdict(list)
        for t in triplets:
            by_slot[t.slot].append(t)

        # Collect all slots that need processing (may have deletions even without new triplets)
        all_slots = set(by_slot.keys()) | set(inline_deletions_by_slot.keys())
        if self.triplet_deletion_mode in ("heuristic", "llm_separate"):
            all_slots |= set(selected_slots)

        for slot in all_slots:
            if slot not in state.slots:
                state.slots[slot] = []

            slot_triplets = by_slot.get(slot, [])

            # Expire slot before processing (lazy expiry)
            expired_in_slot = self._expire_slot(state, slot)
            if expired_in_slot and self.ragu_processor is not None:
                from ..storage.ragu_graph_processor import GraphTripletDelete
                self.ragu_processor.delete_triplet_deltas([
                    GraphTripletDelete(record_id=rid, dialogue_id=dialogue_id, slot=slot)
                    for rid in expired_in_slot
                ])

            # --- DELETION PASS (before semantic dedup and conflict resolution) ---
            # Order: deletions first → then semantic dedup → then conflict → then insert.
            # This ensures conflict resolver sees post-deletion state.

            deletion_signals: List[DeletionSignal] = []

            if self.triplet_deletion_mode == "llm_inline":
                # Вариант A: сигналы уже получены при экстракции с контекстом
                deletion_signals = inline_deletions_by_slot.get(slot, [])

            elif self.triplet_deletion_mode == "heuristic" and self.negation_detector is not None:
                # Вариант C: эвристический детектор отрицания
                active_for_heuristic = [r for r in state.slots[slot] if r.is_active]
                if active_for_heuristic:
                    deletion_signals = self.negation_detector.detect_deletions(
                        user_text, active_for_heuristic
                    )

            elif self.triplet_deletion_mode == "llm_separate" and self.deletion_client is not None:
                # Вариант B: отдельный LLM-вызов, всегда получает текущие факты
                active_for_llm = [r for r in state.slots[slot] if r.is_active]
                if active_for_llm:
                    existing_lines_del = [
                        r.as_line() for r in active_for_llm[: self.slot_context_max_facts]
                    ]
                    deletion_signals = self.deletion_client.detect_deletions(
                        user_text, slot, existing_lines_del
                    )

            if deletion_signals:
                deletion_source_map = {
                    "llm_inline": "llm_inline",
                    "llm_separate": "llm_separate",
                    "heuristic": "heuristic",
                }
                src = deletion_source_map.get(self.triplet_deletion_mode, "unknown")
                self._apply_deletion_signals(dialogue_id, state, slot, deletion_signals, deletion_source=src)
                logger.info(
                    "Applied %d deletion signal(s) slot=%s mode=%s",
                    len(deletion_signals), slot, self.triplet_deletion_mode,
                )

            # --- Semantic dedup PRE-PASS ---
            # For each new triplet: check if a very similar active record exists.
            # If yes: deactivate old, insert new (refreshed TTL timer).
            semantic_dedup_deactivate: List[int] = []
            semantic_dedup_skip_new: set = set()  # do NOT skip — we insert new (replaces old)

            active_in_slot = [r for r in state.slots[slot] if r.is_active]

            if self.semantic_dedup_enabled:
                for idx, new_t in enumerate(slot_triplets):
                    similar_rec = self._semantic_dedup(new_t, active_in_slot)
                    if similar_rec is None or similar_rec.record_id in semantic_dedup_deactivate:
                        continue

                    new_len = len(new_t.as_line())
                    old_len = len(similar_rec.value)

                    if new_len > old_len:
                        # New triplet is richer → replace old record with new
                        semantic_dedup_deactivate.append(similar_rec.record_id)
                        logger.info(
                            "Semantic dedup: replacing record_id=%d (old_len=%d) "
                            "with new triplet (new_len=%d) [%s|%s|%s]",
                            similar_rec.record_id, old_len, new_len,
                            new_t.subject, new_t.relation, new_t.object,
                        )
                    else:
                        # Existing record is longer/richer → keep old, skip new
                        semantic_dedup_skip_new.add(idx)
                        logger.info(
                            "Semantic dedup: keeping record_id=%d (old_len=%d >= new_len=%d), "
                            "skipping new [%s|%s|%s]",
                            similar_rec.record_id, old_len, new_len,
                            new_t.subject, new_t.relation, new_t.object,
                        )

                # Apply semantic dedup deactivations and track deleted facts
                for rid in semantic_dedup_deactivate:
                    for rec in state.slots[slot]:
                        if rec.record_id == rid and rec.is_active:
                            rec.is_active = False
                            rec.updated_at_step = state.step
                            # Track deletion with reason
                            deleted_fact = DeletedFact(
                                slot=slot,
                                record_id=rec.record_id,
                                subject=rec.subject,
                                relation=rec.relation,
                                object=rec.object,
                                value=rec.value,
                                source_text=rec.source_text,
                                created_at_step=rec.created_at_step,
                                created_at_datetime=rec.created_at_datetime,
                                deleted_at_step=state.step,
                                deletion_reason='semantic_dedup',
                                deletion_source='dedup_engine',
                                deletion_details={
                                    'replacement_subject': new_t.subject,
                                    'replacement_relation': new_t.relation,
                                    'replacement_object': new_t.object,
                                }
                            )
                            state.deleted_facts.append(deleted_fact)

                if semantic_dedup_deactivate and self.ragu_processor is not None:
                    from ..storage.ragu_graph_processor import GraphTripletDelete
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

                    # Build record_id → triplet text map for logging
                    _id_to_text = {e.record_id: f"{e.subject}|{e.relation}|{e.object}" for e in all_existing}

                    for rid in resolution.deactivate_ids:
                        for rec in state.slots.get(slot, []):
                            if rec.record_id == rid and rec.is_active:
                                rec.is_active = False
                                rec.updated_at_step = state.step
                                deactivated_record_ids.append(rid)
                                # Track deletion with reason
                                deleted_fact = DeletedFact(
                                    slot=slot,
                                    record_id=rec.record_id,
                                    subject=rec.subject,
                                    relation=rec.relation,
                                    object=rec.object,
                                    value=rec.value,
                                    source_text=rec.source_text,
                                    created_at_step=rec.created_at_step,
                                    created_at_datetime=rec.created_at_datetime,
                                    deleted_at_step=state.step,
                                    deletion_reason='conflict_resolution',
                                    deletion_source='conflict_resolver',
                                    deletion_details={
                                        'conflict_description': f"Conflict with new triplets in slot {slot}"
                                    }
                                )
                                state.deleted_facts.append(deleted_fact)
                                logger.info(
                                    "Conflict resolver: DEACTIVATED record_id=%d slot=%s "
                                    "triplet=[%s] step=%d",
                                    rid, slot,
                                    _id_to_text.get(rid, "?|?|?"),
                                    state.step,
                                )

                    for idx in resolution.skip_new_indices:
                        if 0 <= idx < len(slot_triplets):
                            t = slot_triplets[idx]
                            logger.info(
                                "Conflict resolver: SKIPPED new idx=%d slot=%s "
                                "triplet=[%s|%s|%s]",
                                idx, slot, t.subject, t.relation, t.object,
                            )

                    skip_indices = resolution.skip_new_indices

                    if not resolution.deactivate_ids and not resolution.skip_new_indices:
                        logger.debug(
                            "Conflict resolver: no conflicts found slot=%s "
                            "existing=%d new=%d",
                            slot, len(all_existing), len(slot_triplets),
                        )

            if self.ragu_processor is not None and deactivated_record_ids:
                from ..storage.ragu_graph_processor import GraphTripletDelete
                self.ragu_processor.delete_triplet_deltas([
                    GraphTripletDelete(record_id=rid, dialogue_id=dialogue_id, slot=slot)
                    for rid in deactivated_record_ids
                ])

            # --- Insert surviving new triplets ---
            new_deltas = []
            inserted_count = 0
            for idx, t in enumerate(slot_triplets):
                if idx in skip_indices or idx in semantic_dedup_skip_new:
                    logger.info(
                        "Conflict resolver: SKIPPED insertion idx=%d slot=%s "
                        "triplet=[%s|%s|%s] reason=%s",
                        idx, slot, t.subject, t.relation, t.object,
                        "conflict_resolver" if idx in skip_indices else "semantic_dedup",
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
                    created_at_datetime=self._new_fact_created_at_iso(
                        state, override_iso=fact_created_at_iso,
                    ),
                )
                state.slots[slot].append(rec)

                if self.ragu_processor is not None:
                    from ..storage.ragu_graph_processor import GraphTripletDelta
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
                inserted_count += 1
                logger.info(
                    "INSERTED record_id=%d slot=%s triplet=[%s|%s|%s] ttl=%s step=%d",
                    rid, slot, t.subject, t.relation, t.object, ttl, state.step,
                )

            if self.ragu_processor is not None and new_deltas:
                self.ragu_processor.upsert_triplet_deltas(new_deltas)

            logger.info(
                "Slot %s summary step=%d: inserted=%d skipped=%d deactivated_conflict=%d "
                "deactivated_deletion=%d deactivated_dedup=%d",
                slot, state.step,
                inserted_count,
                len(skip_indices) + len(semantic_dedup_skip_new),
                len(deactivated_record_ids),
                len([s for s in deletion_signals]),
                len(semantic_dedup_deactivate),
            )

        return created, selected_slots

    def deactivate_record(self, dialogue_id: str, record_id: int, reason: str = 'manual') -> bool:
        state = self.get_state(dialogue_id)
        found_slot: Optional[str] = None
        found_rec: Optional[FactRecord] = None
        for slot, records in state.slots.items():
            for rec in records:
                if rec.record_id == record_id and rec.is_active:
                    rec.is_active = False
                    rec.updated_at_step = state.step
                    found_slot = slot
                    found_rec = rec
                    break
            if found_slot is not None:
                break

        if found_slot is None or found_rec is None:
            return False

        # Track deletion with reason
        deleted_fact = DeletedFact(
            slot=found_slot,
            record_id=found_rec.record_id,
            subject=found_rec.subject,
            relation=found_rec.relation,
            object=found_rec.object,
            value=found_rec.value,
            source_text=found_rec.source_text,
            created_at_step=found_rec.created_at_step,
            created_at_datetime=found_rec.created_at_datetime,
            deleted_at_step=state.step,
            deletion_reason=reason,
            deletion_source='manual_call',
        )
        state.deleted_facts.append(deleted_fact)

        if self.ragu_processor is not None:
            from ..storage.ragu_graph_processor import GraphTripletDelete
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
        as_of = self._as_of_datetime_for_state(state)
        result = []
        for slot, records in state.slots.items():
            for rec in records:
                if not rec.is_active and is_expired(rec.ttl, rec.created_at_datetime, as_of=as_of):
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

    def deleted_facts_with_reasons(self, dialogue_id: str) -> List[Dict[str, Any]]:
        """Return all tracked deleted facts with their deletion reasons and sources."""
        state = self.get_state(dialogue_id)
        result = []
        for deleted in state.deleted_facts:
            result.append({
                "slot": deleted.slot,
                "record_id": deleted.record_id,
                "subject": deleted.subject,
                "relation": deleted.relation,
                "object": deleted.object,
                "value": deleted.value,
                "source_text": deleted.source_text,
                "created_at_step": deleted.created_at_step,
                "created_at_datetime": deleted.created_at_datetime,
                "deleted_at_step": deleted.deleted_at_step,
                "deletion_reason": deleted.deletion_reason,
                "deletion_source": deleted.deletion_source,
                "deletion_details": deleted.deletion_details,
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
        Each slot dict includes 'slot_label' (same as canonical English slot id) alongside 'slot'.
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
            result.append({
                "slot": slot,
                "slot_label": slot,
                "messages": messages,
            })
        return result

    def clear_dialogue(self, dialogue_id: str) -> None:
        logger.info("Clearing DST state dialogue_id=%s", dialogue_id)
        self._states.pop(dialogue_id, None)
