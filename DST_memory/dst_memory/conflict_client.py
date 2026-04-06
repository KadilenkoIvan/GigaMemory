"""
Client for LLM-based triplet conflict resolution.

Hybrid strategy:
  1. Rule layer: exact (subject, relation) match → auto-deactivate old record, no LLM call.
  2. LLM layer: same subject, ambiguous/different relation → ask LLM for deactivate/skip_new.

The client is called per-slot after triplet extraction and before insertion.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .conflict_messages import build_conflict_messages, parse_conflict_response
from .graph_backend import GraphEdge
from .serving import GenerationConfig, LocalHFServing
from .triplet_client import ExtractedTriplet

logger = logging.getLogger(__name__)


@dataclass
class ConflictResolution:
    """Result of conflict analysis for one (slot, subject) group."""
    deactivate_ids: List[int]
    skip_new_indices: Set[int]


class TripletConflictClient:
    """
    Resolves conflicts between existing graph triplets and newly extracted ones.

    Rule-based pass runs first (no LLM). LLM is only called when rule pass
    finds same-subject candidates that were NOT handled by exact-match dedup.
    """

    def __init__(
        self,
        *,
        use_stub: bool,
        serving: Optional[LocalHFServing] = None,
        max_retries: int = 1,
    ):
        self.use_stub = use_stub
        self.serving = serving
        self.max_retries = max_retries

        if not use_stub and serving is None:
            raise ValueError("TripletConflictClient requires serving when use_stub is False")

    def resolve(
        self,
        slot_name: str,
        existing_edges: List[GraphEdge],
        new_triplets: List[ExtractedTriplet],
    ) -> ConflictResolution:
        """
        Returns which existing record_ids to deactivate and which new triplet
        indices to skip before insertion.
        """
        deactivate: List[int] = []
        skip_new: Set[int] = set()

        # --- Rule layer: exact (subject, relation) match ---
        handled_existing: Set[int] = set()
        subjects_needing_llm: Set[str] = set()

        for idx, new_t in enumerate(new_triplets):
            exact_matches = [
                e for e in existing_edges
                if e.subject == new_t.subject and e.relation == new_t.relation
                and e.record_id not in handled_existing
            ]
            if exact_matches:
                # Exact duplicate (same object too) → skip new
                if any(e.object == new_t.object for e in exact_matches):
                    skip_new.add(idx)
                    logger.debug(
                        "Rule dedup: skip new idx=%d (%s|%s|%s)",
                        idx, new_t.subject, new_t.relation, new_t.object,
                    )
                else:
                    # Same subject+relation, different object → deactivate old
                    for e in exact_matches:
                        deactivate.append(e.record_id)
                        handled_existing.add(e.record_id)
                    logger.debug(
                        "Rule update: deactivate %s for new (%s|%s|%s)",
                        [e.record_id for e in exact_matches],
                        new_t.subject, new_t.relation, new_t.object,
                    )

        # --- Determine subjects that still need LLM attention ---
        # Any subject in new triplets (not fully handled) that has existing edges
        # with a DIFFERENT relation (semantic ambiguity).
        remaining_new = [
            (idx, t) for idx, t in enumerate(new_triplets)
            if idx not in skip_new
        ]
        for idx, new_t in remaining_new:
            ambiguous_existing = [
                e for e in existing_edges
                if e.subject == new_t.subject
                and e.record_id not in handled_existing
                and e.relation != new_t.relation  # different relation — potential semantic conflict
            ]
            if ambiguous_existing:
                subjects_needing_llm.add(new_t.subject)

        if not subjects_needing_llm or self.use_stub:
            return ConflictResolution(deactivate_ids=deactivate, skip_new_indices=skip_new)

        # --- LLM layer: per-subject semantic conflict check ---
        llm_existing = [
            {"record_id": e.record_id, "subject": e.subject,
             "relation": e.relation, "object": e.object}
            for e in existing_edges
            if e.subject in subjects_needing_llm
            and e.record_id not in handled_existing
        ]
        llm_new = [
            {"idx": idx, "subject": t.subject, "relation": t.relation, "object": t.object}
            for idx, t in remaining_new
            if t.subject in subjects_needing_llm
        ]

        if not llm_existing or not llm_new:
            return ConflictResolution(deactivate_ids=deactivate, skip_new_indices=skip_new)

        llm_result = self._call_llm(slot_name, llm_existing, llm_new)
        if llm_result:
            deactivate.extend(
                rid for rid in llm_result["deactivate"]
                if rid not in handled_existing
            )
            skip_new.update(llm_result["skip_new"])

        return ConflictResolution(deactivate_ids=deactivate, skip_new_indices=skip_new)

    def _call_llm(
        self,
        slot_name: str,
        existing: List[Dict[str, Any]],
        new_triplets: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        assert self.serving is not None
        messages = build_conflict_messages(slot_name, existing, new_triplets)
        cfg = GenerationConfig(max_new_tokens=256, temperature=0.0, do_sample=False)

        for attempt in range(self.max_retries + 1):
            raw = self.serving.generate(messages, cfg)
            last = raw[-1].get("content", "") if raw else ""
            try:
                result = parse_conflict_response(last)
                logger.debug("Conflict LLM resolved slot=%s: %s", slot_name, result)
                return result
            except Exception as exc:
                logger.warning(
                    "Conflict parse failed attempt=%d slot=%s: %s | raw=%r",
                    attempt, slot_name, exc, last[:200],
                )
        return None
