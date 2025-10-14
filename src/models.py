from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    role: str
    content: str
    session_id: Optional[str] = None


def process_first_assistant_message(messages: List[Message]):
    if messages[0].role == "assistant":
        messages = messages[1:]
    return messages


def process_last_user_message(messages: List[Message], placeholder="<нет ответа>"):
    if messages[-1].role == "user":
        messages.append(
            Message(
                session_id=messages[-1].session_id,
                role="assistant",
                content=placeholder,
            )
        )
    return messages


@dataclass
class Session:
    id: str
    messages: List[Message]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        messages_data = data.get("messages", [])
        messages = [
            Message(
                role=message["role"],
                content=message["content"],
                session_id=data.get("id", None),
            )
            for message in messages_data
        ]
        messages = process_first_assistant_message(messages)
        messages = process_last_user_message(messages)
        return cls(id=data["id"], messages=messages)


@dataclass
class Dialog:
    id: str
    sessions: List[Session]
    question: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dialog":
        sessions_data = data.get("sessions", [])
        sessions = [Session.from_dict(session) for session in sessions_data]
        return cls(id=data["id"], sessions=sessions, question=data["question"])

    def get_messages(self) -> List[Message]:
        return [message for session in self.sessions for message in session.messages]


@dataclass
class DialogResponse:
    id: str
    question: str
    pred_answer: str
    success: bool
    error: Optional[str] = None
    answer_time: Optional[float] = None
