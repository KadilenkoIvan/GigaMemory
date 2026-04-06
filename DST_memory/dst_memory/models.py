from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    role: str
    content: str
    session_id: Optional[str] = None


@dataclass
class QATurn:
    question: str
    answer: str


@dataclass
class FactRecord:
    record_id: int
    value: str
    source_text: str
    created_at_step: int
    updated_at_step: int
    is_active: bool = True
    triplets: List[Dict[str, str]] = field(default_factory=list)
    graph_artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryFact:
    slot: str
    record_id: int
    value: str
    source_text: str
    created_at_step: int
    updated_at_step: int
    is_active: bool
    triplets: List[Dict[str, str]] = field(default_factory=list)
    graph_artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogueMemoryState:
    dialogue_id: str
    step: int = 0
    slots: Dict[str, List[FactRecord]] = field(default_factory=dict)
    next_record_id: int = 1
    qa_turns: List[QATurn] = field(default_factory=list)
