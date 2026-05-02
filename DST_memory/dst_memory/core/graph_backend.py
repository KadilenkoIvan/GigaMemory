from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphEdge:
    edge_id: int
    slot: str
    subject: str
    relation: str
    object: str
    record_id: int
    is_active: bool = True
