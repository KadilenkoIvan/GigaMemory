from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
    is_active: bool = True


@dataclass
class MemoryFact:
    slot: str
    record_id: int
    value: str
    source_text: str
    created_at_step: int
    updated_at_step: int
    is_active: bool


@dataclass
class DialogueMemoryState:
    dialogue_id: str
    step: int = 0
    slots: Dict[str, List[FactRecord]] = field(default_factory=dict)
    next_record_id: int = 1
