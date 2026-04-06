from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


@dataclass(frozen=True)
class GraphTripletDelta:
    """Represents an upsert (add/update) delta for a single triplet."""
    dialogue_id: str
    slot: str
    subject: str
    relation: str
    object: str
    record_id: int


@dataclass(frozen=True)
class GraphTripletDelete:
    """Represents a delete delta — the record was superseded or removed."""
    dialogue_id: str
    record_id: int


class GraphProcessor(Protocol):
    """
    Abstraction point for future graph processors (e.g., RAGU).
    Both upserts and deletes are forwarded here so RAGU can do
    partial-graph rebuilds on only the affected subgraphs.
    """

    async def upsert_triplet_deltas(self, deltas: List[GraphTripletDelta]) -> None:
        ...

    async def delete_triplet_deltas(self, deltas: List[GraphTripletDelete]) -> None:
        ...


class NoopGraphProcessor:
    async def upsert_triplet_deltas(self, deltas: List[GraphTripletDelta]) -> None:
        _ = deltas

    async def delete_triplet_deltas(self, deltas: List[GraphTripletDelete]) -> None:
        _ = deltas

