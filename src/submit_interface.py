from abc import ABC, abstractmethod
from typing import List

from models import Message


class ModelWithMemory(ABC):

    @abstractmethod
    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        pass

    @abstractmethod
    def clear_memory(self, dialogue_id: str) -> None:
        pass

    @abstractmethod
    def answer_to_question(self, dialogue_id: str, question: str) -> str:
        pass
