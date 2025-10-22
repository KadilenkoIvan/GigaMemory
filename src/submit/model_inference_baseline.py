from collections import defaultdict
from dataclasses import asdict
from typing import List

from transformers import AutoTokenizer
#from vllm import LLM, SamplingParams

from models import Message
from submit_interface import ModelWithMemory


class SubmitModelWithMemory(ModelWithMemory):

    def __init__(self, model_path: str) -> None:
        self.basic_memory = defaultdict(list)
        self.model_path = model_path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = LLM(model=self.model_path, trust_remote_code=True)
            self.sampling_params = SamplingParams(temperature=0.0, max_tokens=100, seed=42, truncate_prompt_tokens=131072)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели {self.model_path}: {str(e)}")

    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        # Save only user messages
        user_messages = [msg for msg in messages if msg.role == "user"] # ИЗМЕНЕНО
        self.basic_memory[dialogue_id] += user_messages # ИЗМЕНЕНО

    def extract(self, dialogue_id: str) -> List[Message]:
        memory = self.basic_memory.get(dialogue_id, [])
        memory = [asdict(msg) for msg in memory]
        memory = "\n".join([f"{msg['role']}: {msg['content']}" for msg in memory])
        system_memory_prompt = "Твоя задача - ответить на вопрос пользователя. Для этого тебе подается на вход твоя история общения с пользователем." \
                               "Пользователь разрешил использовать ее для ответа на вопрос. Используй историю диалога, чтобы ответить на вопрос.\n" \
                               f"История диалога: \n{memory}"

        context_with_memory = [Message('system', system_memory_prompt)]

        return context_with_memory

    def clear_memory(self, dialogue_id: str) -> None:
        self.basic_memory[dialogue_id] = []

    def answer_to_question(self, dialogue_id: str, question: str) -> str:
        user_promt = "Отвечай КРАТКО без предложений и размышлений, используй ТОЛЬКО ОДНО предложение для ответа.\n"
        context = self.extract(dialogue_id)
        context.append(Message(
            role="user",
            content=user_promt + question
        ))
        answer = self._inference(context)

        return answer

    def _inference(self, messages: List[Message]) -> str:
        try:
            msg_dicts = [asdict(m) for m in messages]
            input_tensor = self.tokenizer.apply_chat_template(
                msg_dicts,
                add_generation_prompt=True,
            )
            outputs = self.model.generate(prompt_token_ids=input_tensor, sampling_params=self.sampling_params, use_tqdm=False)
            result = outputs[0].outputs[0].text
            return result.strip()

        except Exception as e:
            return f"Ошибка при инференсе локальной модели: {str(e)}"
