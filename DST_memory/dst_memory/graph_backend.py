from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple


@dataclass
class GraphEdge:
    edge_id: int
    slot: str
    subject: str
    relation: str
    object: str
    record_id: int
    is_active: bool = True


@dataclass
class UserGraphState:
    nodes: Set[str] = field(default_factory=set)
    edges: Dict[int, GraphEdge] = field(default_factory=dict)  # key: record_id
    adjacency: Dict[str, Set[str]] = field(default_factory=dict)
    slot_to_entities: Dict[str, Set[str]] = field(default_factory=dict)


class UserGraphBackend:
    """
    Lightweight in-memory user graph.
    Designed as a compatibility bridge for later RAGU integration:
    - slot subgraphs are explicit via slot_to_entities
    - triplets are explicit edges
    - supports scoped entity expansion for retrieval
    """

    def __init__(self) -> None:
        self._states: Dict[str, UserGraphState] = {}

    def get_state(self, dialogue_id: str) -> UserGraphState:
        st = self._states.get(dialogue_id)
        if st is None:
            st = UserGraphState()
            self._states[dialogue_id] = st
        return st

    def upsert_triplet(
        self,
        dialogue_id: str,
        *,
        record_id: int,
        slot: str,
        subject: str,
        relation: str,
        object_: str,
    ) -> None:
        st = self.get_state(dialogue_id)
        edge = GraphEdge(
            edge_id=record_id,
            slot=slot,
            subject=subject,
            relation=relation,
            object=object_,
            record_id=record_id,
            is_active=True,
        )
        st.edges[record_id] = edge
        st.nodes.add(subject)
        st.nodes.add(object_)
        st.adjacency.setdefault(subject, set()).add(object_)
        st.adjacency.setdefault(object_, set()).add(subject)
        st.slot_to_entities.setdefault(slot, set()).update((subject, object_))

    def entities_for_slots(self, dialogue_id: str, slot_names: Iterable[str]) -> List[str]:
        st = self.get_state(dialogue_id)
        out: Set[str] = set()
        for s in slot_names:
            out.update(st.slot_to_entities.get(s, set()))
        return sorted(out)

    def expand_entities(self, dialogue_id: str, seeds: Iterable[str], hops: int = 1) -> List[str]:
        st = self.get_state(dialogue_id)
        visited: Set[str] = set(str(x) for x in seeds if x)
        frontier: Set[str] = set(visited)
        for _ in range(max(0, hops)):
            nxt: Set[str] = set()
            for n in frontier:
                nxt.update(st.adjacency.get(n, set()))
            nxt -= visited
            if not nxt:
                break
            visited.update(nxt)
            frontier = nxt
        return sorted(visited)

    def slot_subgraph_edges(self, dialogue_id: str, slot_names: Iterable[str]) -> List[GraphEdge]:
        slots = set(slot_names)
        st = self.get_state(dialogue_id)
        return [e for e in st.edges.values() if e.is_active and e.slot in slots]

    def clear_dialogue(self, dialogue_id: str) -> None:
        self._states.pop(dialogue_id, None)

