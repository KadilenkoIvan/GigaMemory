import json
from typing import Dict, Iterable, List

from .models import Message


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def iter_user_messages(record: Dict) -> Iterable[Message]:
    for session in record.get("sessions", []):
        sid = session.get("id")
        for msg in session.get("messages", []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and isinstance(content, str):
                yield Message(role="user", content=content, session_id=sid)


def iter_dialogue_messages(record: Dict) -> Iterable[Message]:
    for session in record.get("sessions", []):
        sid = session.get("id")
        for msg in session.get("messages", []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and isinstance(content, str):
                yield Message(role=role, content=content, session_id=sid)
