"""
Client for LLM-based triplet conflict resolution.

Hybrid strategy:
  1. Rule layer: exact (subject, relation) duplicate (same object) → skip new; optionally
     same subject+relation with different object → auto-deactivate old (`rule_same_relation_updates`).
  2. LLM layer: unresolved same-subject groups → ask LLM for deactivate/skip_new.

The client is called per-slot after triplet extraction and before insertion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ..clients.serving import GenerationConfig, SlotServing
from ..core.graph_backend import GraphEdge
from ..prompts.loader import PromptModules, load_prompt_modules
from ..prompts.parsers import parse_conflict_response
from .triplet_client import ExtractedTriplet

logger = logging.getLogger(__name__)


@dataclass
class ConflictResolution:
    """Result of conflict analysis for one (slot, subject) group."""

    deactivate_ids: list[int]
    skip_new_indices: set[int]


class TripletConflictClient:
    """
    Resolves conflicts between existing graph triplets and newly extracted ones.

    Rule-based pass runs first (no LLM unless deferred). LLM runs when rule pass leaves
    same-subject groups unresolved (different relation ambiguity, or same relation + different
    object when rule_same_relation_updates=False).

    rule_same_relation_updates:
        When True (default): same subject AND same relation but different object is treated as a
        value replacement — existing matching edges are auto-deactivated (no LLM).
        When False: that case is deferred to the LLM conflict call only (duplicate same S+R+O still
        skips new via rules).

    allow_multi_relation_same_object:
        When True (default), two triplets with the same subject AND the same object
        but DIFFERENT relations are treated as complementary facts and the LLM
        conflict check is skipped for them.
        Example: "пользователь | есть партнёр | партнёр пользователя" and
                 "пользователь | живёт вместе с | партнёр пользователя"
                 → both are kept as separate, valid facts.
        Set to False to always run the LLM check for any same-subject pair.
    """

    def __init__(
        self,
        *,
        use_stub: bool,
        serving: SlotServing | None = None,
        max_retries: int = 1,
        rule_same_relation_updates: bool = True,
        allow_multi_relation_same_object: bool = True,
        prompt_language: str = "ru",
        max_new_tokens: int = 256,
    ):
        self.use_stub = use_stub
        self.serving = serving
        self.max_retries = max_retries
        self.rule_same_relation_updates = rule_same_relation_updates
        self.allow_multi_relation_same_object = allow_multi_relation_same_object
        self.prompt_language = prompt_language
        self.max_new_tokens = int(max_new_tokens)
        self._prompt_modules: PromptModules | None = None

        if not use_stub and serving is None:
            raise ValueError(
                "TripletConflictClient requires serving when use_stub is False"
            )

    def resolve(
        self,
        slot_name: str,
        existing_edges: list[GraphEdge],
        new_triplets: list[ExtractedTriplet],
    ) -> ConflictResolution:
        """
        Returns which existing record_ids to deactivate and which new triplet
        indices to skip before insertion.
        """
        deactivate: list[int] = []
        skip_new: set[int] = set()

        # --- Rule layer: exact (subject, relation) match ---
        handled_existing: set[int] = set()
        subjects_needing_llm: set[str] = set()

        for idx, new_t in enumerate(new_triplets):
            exact_matches = [
                e
                for e in existing_edges
                if e.subject == new_t.subject
                and e.relation == new_t.relation
                and e.record_id not in handled_existing
            ]
            if exact_matches:
                # Exact duplicate (same object too) → skip new
                if any(e.object == new_t.object for e in exact_matches):
                    skip_new.add(idx)
                    logger.debug(
                        "Rule dedup: skip new idx=%d (%s|%s|%s)",
                        idx,
                        new_t.subject,
                        new_t.relation,
                        new_t.object,
                    )
                else:
                    # Same subject+relation, different object
                    if self.rule_same_relation_updates:
                        for e in exact_matches:
                            deactivate.append(e.record_id)
                            handled_existing.add(e.record_id)
                        logger.debug(
                            "Rule update: deactivate %s for new (%s|%s|%s)",
                            [e.record_id for e in exact_matches],
                            new_t.subject,
                            new_t.relation,
                            new_t.object,
                        )
                    else:
                        subjects_needing_llm.add(new_t.subject)
                        logger.debug(
                            "Conflict deferred to LLM (same S+R, different O): subj=%s rel=%s "
                            "new_obj=%r existing_ids=%s",
                            new_t.subject,
                            new_t.relation,
                            new_t.object,
                            [e.record_id for e in exact_matches],
                        )

        # --- Determine subjects that still need LLM attention ---
        # Any subject in new triplets (not fully handled) that has existing edges
        # with a DIFFERENT relation (semantic ambiguity).
        remaining_new = [
            (idx, t) for idx, t in enumerate(new_triplets) if idx not in skip_new
        ]
        for idx, new_t in remaining_new:
            ambiguous_existing = [
                e
                for e in existing_edges
                if e.subject == new_t.subject
                and e.record_id not in handled_existing
                and e.relation
                != new_t.relation  # different relation — potential semantic conflict
            ]
            if not ambiguous_existing:
                continue

            if self.allow_multi_relation_same_object:
                # Filter out complementary facts: same subject + same object + different relation.
                # These add independent information and should NOT trigger conflict resolution.
                truly_ambiguous = [
                    e
                    for e in ambiguous_existing
                    if e.object
                    != new_t.object  # different object → potentially conflicting
                ]
                if truly_ambiguous:
                    subjects_needing_llm.add(new_t.subject)
                    logger.debug(
                        "Conflict LLM candidate: subj=%s new_rel=%s (complementary same-object "
                        "excluded=%d, ambiguous=%d)",
                        new_t.subject,
                        new_t.relation,
                        len(ambiguous_existing) - len(truly_ambiguous),
                        len(truly_ambiguous),
                    )
                else:
                    logger.debug(
                        "Conflict: all ambiguous existing have same object as new triplet "
                        "→ treating as complementary, skipping LLM for subj=%s",
                        new_t.subject,
                    )
            else:
                subjects_needing_llm.add(new_t.subject)

        if not subjects_needing_llm or self.use_stub:
            return ConflictResolution(
                deactivate_ids=deactivate, skip_new_indices=skip_new
            )

        # --- LLM layer: per-subject semantic conflict check ---
        llm_existing = [
            {
                "record_id": e.record_id,
                "subject": e.subject,
                "relation": e.relation,
                "object": e.object,
            }
            for e in existing_edges
            if e.subject in subjects_needing_llm and e.record_id not in handled_existing
        ]
        llm_new = [
            {
                "idx": idx,
                "subject": t.subject,
                "relation": t.relation,
                "object": t.object,
            }
            for idx, t in remaining_new
            if t.subject in subjects_needing_llm
        ]

        if not llm_existing or not llm_new:
            return ConflictResolution(
                deactivate_ids=deactivate, skip_new_indices=skip_new
            )

        llm_result = self._call_llm(slot_name, llm_existing, llm_new)
        if llm_result:
            deactivate.extend(
                rid for rid in llm_result["deactivate"] if rid not in handled_existing
            )
            skip_new.update(llm_result["skip_new"])

        return ConflictResolution(deactivate_ids=deactivate, skip_new_indices=skip_new)

    def _call_llm(
        self,
        slot_name: str,
        existing: list[dict[str, Any]],
        new_triplets: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        assert self.serving is not None
        if self._prompt_modules is None:
            self._prompt_modules = load_prompt_modules(self.prompt_language)
        messages = self._prompt_modules.conflict_messages.build_conflict_messages(
            slot_name, existing, new_triplets
        )
        cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens, temperature=0.0, do_sample=False
        )

        for attempt in range(self.max_retries + 1):
            last = self.serving.generate_chat(messages, cfg)
            try:
                result = parse_conflict_response(last)
                logger.debug("Conflict LLM resolved slot=%s: %s", slot_name, result)
                return result
            except Exception as exc:
                logger.warning(
                    "Conflict parse failed attempt=%d slot=%s: %s | raw=%r",
                    attempt,
                    slot_name,
                    exc,
                    last[:200],
                )
        return None
