import time
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional

from ragu.utils.ragu_utils import FLOATS, compute_mdhash_id, serialize


class Node:
    """
    Base graph node type for storage adapters.

    Subclasses are expected to be dataclasses and define an ``id`` field.
    """

    id: str

    def to_dict(self) -> Dict[str, Any]:
        return serialize(self)


class Edge:
    """
    Base graph edge type for storage adapters.

    Subclasses are expected to be dataclasses and define ``id``,
    ``subject_id``, and ``object_id`` fields.
    """

    id: str
    subject_id: str
    object_id: str

    def to_dict(self) -> Dict[str, Any]:
        return serialize(self)


@dataclass(slots=True)
class Embedding:
    """
    Representation of an embedding.

    :param id: Unique record identifier.
    :param vector: Embedding vector.
    :param metadata: Additional payload.
    """
    vector: List[float] | FLOATS
    metadata: Dict[str, Any] = field(default_factory=dict[str, Any])
    id: Optional[str] = None

    def __post_init__(self):
        # If id is not set, generate a random one
        if self.id is None:
            self.id = compute_mdhash_id(str(time.time_ns()), prefix="emb")


@dataclass(slots=True)
class EmbeddingHit:
    """
    Vector query hit.

    :param id: Matched record identifier.
    :param distance: Similarity/distance score to query embedding.
    :param metadata: Additional payload.
    """
    id: str
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict[str, Any])
