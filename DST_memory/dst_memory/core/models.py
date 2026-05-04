from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

VALID_TTL_VALUES = {"1d", "3d", "10d", "2w", "3w", "1m", "3m", "6m", "1y", "inf"}

TTL_TO_TIMEDELTA: Dict[str, Optional[timedelta]] = {
    "1d": timedelta(days=1),
    "3d": timedelta(days=3),
    "10d": timedelta(days=10),
    "2w": timedelta(weeks=2),
    "3w": timedelta(weeks=3),
    "1m": timedelta(days=30),
    "3m": timedelta(days=90),
    "6m": timedelta(days=180),
    "1y": timedelta(days=365),
    "inf": None,
}


def is_expired(
    ttl: str,
    created_at_datetime: str,
    as_of: Optional[datetime] = None,
) -> bool:
    """Return True if TTL has elapsed since ``created_at_datetime`` relative to ``as_of`` (default: real now)."""
    if ttl == "inf" or not created_at_datetime:
        return False
    delta = TTL_TO_TIMEDELTA.get(ttl)
    if delta is None:
        return False
    try:
        created = datetime.fromisoformat(created_at_datetime)
    except ValueError:
        return False
    ref = as_of if as_of is not None else datetime.now()
    return ref > created + delta


def now_iso() -> str:
    return datetime.now().isoformat()


@dataclass
class Message:
    role: str
    content: str
    session_id: Optional[str] = None


@dataclass
class FactRecord:
    record_id: int
    value: str
    source_text: str
    created_at_step: int
    updated_at_step: int
    subject: str = ""
    relation: str = ""
    object: str = ""
    is_active: bool = True
    # TTL fields
    ttl: str = "inf"
    created_at_datetime: str = field(default_factory=now_iso)

    def is_expired(self, as_of: Optional[datetime] = None) -> bool:
        return is_expired(self.ttl, self.created_at_datetime, as_of=as_of)

    def refresh_ttl(self) -> None:
        """Reset the TTL timer (called on semantic dedup match)."""
        self.created_at_datetime = now_iso()

    def as_line(self) -> str:
        """Compact triplet string used for slot context passed to LLM."""
        return f"{self.subject} | {self.relation} | {self.object}"


@dataclass
class MemoryFact:
    slot: str
    record_id: int
    value: str
    source_text: str
    created_at_step: int
    updated_at_step: int
    is_active: bool
    subject: str = ""
    relation: str = ""
    object: str = ""
    ttl: str = "inf"
    created_at_datetime: str = ""


@dataclass
class DeletedFact:
    """Record of a deleted/soft-deleted fact with deletion metadata."""
    slot: str
    record_id: int
    subject: str
    relation: str
    object: str
    value: str
    source_text: str
    created_at_step: int
    created_at_datetime: str
    deleted_at_step: int
    deletion_reason: str  # 'ttl_expired', 'deletion_signal', 'conflict_resolution', 'semantic_dedup', 'manual'
    deletion_source: str  # 'llm_inline', 'llm_separate', 'heuristic', 'ttl_checker', 'conflict_resolver', 'dedup_engine'
    deletion_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogueMemoryState:
    dialogue_id: str
    step: int = 0
    slots: Dict[str, List[FactRecord]] = field(default_factory=dict)
    next_record_id: int = 1
    recent_pairs: List[Dict[str, str]] = field(default_factory=list)
    deleted_facts: List[DeletedFact] = field(default_factory=list)  # Track all soft-deleted facts with reasons
    # LongMemEval validation: ``question_date`` as ISO — TTL lazy expiry "as_of" + final-LLM clock.
    # Per-fact insert times may differ (``haystack_dates[i]``) via ``upsert_from_message(..., fact_created_at_iso=)``.
    dataset_clock_iso: Optional[str] = None
