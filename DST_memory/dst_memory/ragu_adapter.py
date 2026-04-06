from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


@dataclass(frozen=True)
class GraphTripletDelta:
    dialogue_id: str
    slot: str
    subject: str
    relation: str
    object: str
    record_id: int


class GraphProcessor(Protocol):
    """
    Abstraction point for future graph processors (e.g., RAGU).
    """

    async def upsert_triplet_deltas(self, deltas: List[GraphTripletDelta]) -> None:
        ...


class NoopGraphProcessor:
    async def upsert_triplet_deltas(self, deltas: List[GraphTripletDelta]) -> None:
        _ = deltas
        return None

