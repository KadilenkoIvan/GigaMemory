"""
JSON Schema dicts for lm-format-enforcer (slot selector + triplet extraction).

Kept minimal and aligned with SlotSelectClient / TripletExtractionClient parsers.
"""
from __future__ import annotations

from typing import Any, Dict

# --- Slot selection: {"slot_assignments": ["WORK", ...]} ---
SLOT_SELECT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "slot_assignments": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["slot_assignments"],
    "additionalProperties": False,
}

_TRIPLET_ITEM: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "slot": {"type": "string"},
        "subject": {"type": "string"},
        "relation": {"type": "string"},
        "object": {"type": "string"},
        "ttl": {"type": "string"},
    },
    "required": ["slot", "subject", "relation", "object"],
    "additionalProperties": False,
}

_DELETE_ITEM: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "relation": {"type": "string"},
        "object": {"type": "string"},
    },
    "required": ["subject", "relation", "object"],
    "additionalProperties": False,
}

# Triplet extraction without inline deletion signals.
TRIPLET_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "triplets": {"type": "array", "items": _TRIPLET_ITEM},
    },
    "required": ["triplets"],
    "additionalProperties": False,
}

# Same as above but allows optional "delete" list (llm_inline deletion mode).
TRIPLET_JSON_SCHEMA_WITH_DELETE: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "triplets": {"type": "array", "items": _TRIPLET_ITEM},
        "delete": {"type": "array", "items": _DELETE_ITEM},
    },
    "required": ["triplets"],
    "additionalProperties": False,
}
