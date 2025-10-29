import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum


# def _bootstrap_offline_dependencies() -> None:
#     """Bootstrap offline dependencies from libs directory."""
#     os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
#     os.environ.setdefault("HF_HUB_OFFLINE", "1")
    
#     current_dir = Path(__file__).resolve().parent  # DST/
#     candidate_dirs = [
#         current_dir / "libs",
#         current_dir.parent / "libs",  # submit/libs
#         current_dir.parent.parent / "libs",  # src/libs
#     ]
#     libs_dir = next((d for d in candidate_dirs if d.exists() and d.is_dir()), None)
#     if not libs_dir:
#         return
    
#     archives = list(libs_dir.glob("*.whl")) + list(libs_dir.glob("*.tar.gz"))
#     if not archives:
#         return
    
#     for pkg in sorted(archives):
#         try:
#             subprocess.run(
#                 [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(libs_dir), str(pkg)],
#                 check=True,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#             )
#         except Exception:
#             pass


# _bootstrap_offline_dependencies()

from transformers import AutoTokenizer
import torch

# Import Message from parent directory
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]  # Go up to src/
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from models import Message


class DSTAction(Enum):
    """Actions that DST can decide for a message"""
    NO_ACTION = "NO_ACTION"  # No new information
    UPDATE = "UPDATE"  # New information to add
    DELETE_AND_UPDATE = "DELETE_AND_UPDATE"  # Update existing information or mark as outdated


class DSTProcessor:
    """
    Dialog State Tracking processor that determines if a user message should be saved to memory.
    Uses local quantized Mistral-7B model for decision making and fact summarization.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the DST processor with Qwen3-1.7B model.
        
        Args:
            model_path: Path to the model. If None, will try to use default path.
        """
        if model_path is None:
            # Try to find model in default location
            base_dir = Path(__file__).resolve().parent
            model_path = base_dir / "../models" / "Qwen3-1.7B"
        
        self.model_path = str(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Load Qwen3 model with transformers
            from transformers import AutoModelForCausalLM
            
            print(f"🔍 CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"🔍 CUDA device: {torch.cuda.get_device_name(0)}")
                print(f"🔍 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            print(f"Loading Qwen3 model from: {self.model_path}")
            print(f"Target device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None  # Automatically use GPU
            )
            
            # Verify model is on GPU
            if self.device == "cuda":
                print(f"✅ Model loaded on device: {next(self.model.parameters()).device}")
            else:
                print(f"⚠️ Model loaded on CPU")
            
            self.is_gguf = False
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели DST {self.model_path}: {str(e)}")
    
    def should_save_message(self, message: Message) -> Tuple[bool, str]:
        """
        Determine if a message should be saved to memory (legacy method for compatibility).
        
        Args:
            message: The message to evaluate
            
        Returns:
            Tuple of (should_save, reason)
            - should_save: True if message should be saved, False otherwise
            - reason: Explanation of the decision
        """
        action, reason = self.determine_action(message.content, memory_summary="")
        should_save = (action != DSTAction.NO_ACTION)
        return should_save, reason
    
    def determine_action(self, user_message: str, memory_summary: str) -> Tuple[DSTAction, str]:
        """
        Determine what action to take with the user message based on existing memory.
        
        Args:
            user_message: The content of the user message
            memory_summary: Summary of existing memory about the user
            
        Returns:
            Tuple of (action, reason)
            - action: DSTAction (NO_ACTION, UPDATE, DELETE_AND_UPDATE)
            - reason: Explanation of the decision
        """
        # Construct prompts with memory context
        prompts = self._construct_action_prompt(user_message, memory_summary)
        
        # Get model prediction
        response = self._get_model_response(prompts, max_tokens=10)
        
        # Parse the response to determine action
        action, reason = self._parse_action_response(response)
        
        return action, reason
    
    def extract_facts(self, message: Message) -> Dict[str, List[str]]:
        """
        Extract structured facts from a user message.
        
        Args:
            message: The message to extract facts from
            
        Returns:
            Dictionary with extracted facts in format {category: [values]}
            Example: {"спорт": ["футбол"], "хобби": ["рисование"], "возраст": ["25"]}
        """
        
        # Construct prompts for fact extraction
        prompts = self._construct_extraction_prompt(message.content)
        
        # Get model response
        response = self._get_model_response(prompts, max_tokens=100)
        
        # Parse the response into structured format
        facts = self._parse_facts(response)
        
        return facts
    
    def _construct_prompt(self, message_content: str) -> Tuple[str, str]:
        """
        Construct system and user prompts for the DST model.
        
        Args:
            message_content: The content of the user message
            
        Returns:
            A tuple of (system_prompt, user_prompt)
        """
        system_prompt = """Ты — модуль управления памятью диалогового агента. 
                    Твоя задача — анализировать каждую реплику пользователя и решать, стоит ли сохранять её как важную информацию о пользователе.

                    Действия:
                    - "true" — если реплика содержит информацию о пользователе: его интересы, хобби, род занятий, спортивные увлечения, место проживания, возраст, привычки, вредные привычки, намерения, планы, факты о себе, возможные будущие действия. Даже если это выражено неявно, например: "О, это хороший совет, попробую взять его с собой на футбол." это важно, т.к. пользователь занимается футболом. 
                    - "false" — если реплика не содержит информации о пользователе (например, пустые фразы, шутки, общие вопросы, комментарии без фактов).

                    Инструкции:
                    1. Всегда отвечай только одним словом: true или false.
                    2. Не добавляй пояснений, комментариев или текста.
                    3. Игнорируй эмоциональные реакции, шутки и общие вопросы.

                    Примеры:

                    Пользователь: "Я футболист.", Твой ответ: "true"  
                    Пользователь: "Как дела?", Твой ответ: "false"
                    Пользователь: "Завтра пойду на пробежку", Твой ответ: "true" 
                    Пользователь: "О, это хороший совет, попробую взять его с собой на футбол.", Твой ответ: "true"   
                    Пользователь: "У меня муха на столе!", Твой ответ: "false"
                    """
        
        user_prompt = f"Реплка пользователя: \"{message_content}\""
        
        return system_prompt, user_prompt
    
    def _get_model_response(self, prompts: Tuple[str, str], max_tokens=10) -> str:
        """
        Get response from the model for given system and user prompts.
        
        Args:
            prompts: Tuple of (system_prompt, user_prompt)
            max_tokens: Maximum tokens to generate
            
        Returns:
            The model's response as a string
        """
        system_prompt, user_prompt = prompts
        
        # Qwen3 using transformers
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # Токенизируем сообщения в формате чата
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Check GPU memory before inference
        if self.device == "cuda":
            print(f"🔍 GPU memory before inference: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,    # Greedy decoding
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Check GPU memory after inference
        if self.device == "cuda":
            print(f"🔍 GPU memory after inference: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"model answer-🤖: {response}")
        return response.strip()
    
    def _parse_response(self, response: str) -> Tuple[bool, str]:
        """
        Parse the model's response to determine if message should be saved.
        
        Args:
            response: The model's response
            
        Returns:
            Tuple of (should_save, reason)
        """
        print(f"model answer-🤖: {response}")
        response = response.lower().strip()
        
        if "true" in response:
            return True, "Message contains important information"
        else:
            return False, "Message does not contain important information"
    
    def _construct_action_prompt(self, user_message: str, memory_summary: str) -> Tuple[str, str]:
        system_prompt = """Ты — модуль управления памятью диалогового агента. 
                        Твоя задача — анализировать каждую реплику пользователя и решать, стоит ли сохранять её как важную информацию о пользователе.

                        ВАЖНО: Отвечай ТОЛЬКО одним из двух слов. НЕ добавляй пояснений, размышлений или дополнительного текста!

                        Возможные действия:
                        1. UPDATE — если реплика содержит информацию о пользователе: его интересы, имена, упоминания друзей или родственников, хобби, род занятий, спортивные увлечения, имущество пользователя, его машина, место проживания, возраст, привычки, вредные привычки, намерения, планы, факты о себе, возможные будущие действия, пользователь перестал чем-то заниматься, поменял увлечения, уволился с работы и т.д. Даже если это выражено неявно.
                        2. NO_ACTION — в сообщении НЕТ новой информации о пользователе (вопросы, эмоции, благодарности, общие фразы, запрос на выполнение задачи, без упоминания личной информации).

                        Информация может быть косвенной, например ползователь говорит: "О, хорошая идея! попробую взять его с собой на футбол.", это важно, т.к. пользователь занимается футболом.
                        
                        Также ты видишь последние сообщения пользователя. Если информация в них повторяется, например в обоих упоминается что у пользователя есть кот, И БОЛЬШЕ НЕТ НОВОЙ ИНФОРМАЦИИ, то это не новая информация. 
                        НО, если помимо повторения старой информации, есть новая, то это новая информация, нужно сохранить.

                        Примеры:

                        Сообщение: "Я играю в футбол"
                        Ответ: UPDATE

                        Сообщение: "Ещё занимаюсь баскетболом"
                        Ответ: UPDATE

                        Сообщение: "Работаю программистом"
                        Ответ: UPDATE

                        Сообщение: "Завтра пойду на пробежку"
                        Ответ: UPDATE

                        Сообщение: "У меня муха на стол залетела, отвлекает"
                        Ответ: NO_ACTION

                        Сообщение: "Как дела?"
                        Ответ: NO_ACTION
                        """ + f"\n\nПоследние сообщения пользователя: {memory_summary}"
        
        # Inject up to 5 last saved user messages as compact context (can be empty)
        context_block = (
            f"Последние сообщения пользователя:\n{memory_summary}\n\n"
            if memory_summary.strip() else ""
        )
        user_prompt = (
            context_block +
            f"Новое сообщение пользователя:\n\"{user_message}\""
        )
        
        return system_prompt, user_prompt
    
    def _parse_action_response(self, response: str) -> Tuple[DSTAction, str]:
        """
        Parse model response to determine action.
        
        Args:
            response: The model's response
            
        Returns:
            Tuple of (action, reason)
        """
        print(f"DST action-🎯: {response}")
        response_clean = response.upper().strip()

        if "UPDATE" in response_clean:
            return DSTAction.UPDATE, "New information to add"
        elif "NO_ACTION" in response_clean or "NO" in response_clean:
            return DSTAction.NO_ACTION, "No new information"
        else:
            # Default to NO_ACTION if unclear
            print(f"Warning: Unclear DST response '{response}', defaulting to NO_ACTION")
            return DSTAction.NO_ACTION, "Unclear response"
    
    def _construct_extraction_prompt(self, message_content: str) -> Tuple[str, str]:
        """
        Construct prompts for extracting structured facts from user message.
        
        Args:
            message_content: The content of the user message
            
        Returns:
            A tuple of (system_prompt, user_prompt)
        """
        system_prompt = """Ты — модуль извлечения структурированной информации из реплик пользователя.
                    Твоя задача — вычленить важную информацию о пользователе и представить её в формате "категория: значение".

                    Категории информации:
                    - интересы: увлечения и интересы пользователя
                    - хобби: занятия в свободное время
                    - спорт: спортивные увлечения
                    - работа: профессия, род занятий
                    - место_жительства: город, страна
                    - возраст: возраст пользователя
                    - привычки: регулярные действия, привычки
                    - планы: намерения, будущие действия
                    - факты: любые другие важные факты о пользователе
                    - имя: имя пользователя
                    - семья: информация о семье (жена, муж, дети и т.д.)
                    - питомцы: домашние животные
                    - образование: учебные заведения, специальность
                    - навыки: умения, навыки
                    - предпочтения: предпочтения в еде, музыке и т.д.

                    Инструкции:
                    1. Извлекай ТОЛЬКО явную информацию из реплики.
                    2. Формат ответа: каждая категория с новой строки в формате "категория: значение1, значение2".
                    3. Если в реплике несколько категорий, выведи все.
                    4. Если категория не подходит, используй "факты".
                    5. Не добавляй пояснений, только факты в указанном формате.

                    Примеры:

                    Пользователь: "Я футболист."
                    Твой ответ:
                    спорт: футбол

                    Пользователь: "Классно! Возьму его с собой на футбол"
                    Твой ответ:
                    спорт: футбол

                    Пользователь: "Завтра пойду на пробежку, а потом встречусь с друзьями"
                    Твой ответ:
                    спорт: бег
                    планы: встреча с друзьями

                    Пользователь: "Меня зовут Иван, я живу в Москве и работаю программистом"
                    Твой ответ:
                    имя: Иван
                    место_жительства: Москва
                    работа: программист

                    Пользователь: "У меня есть кот Барсик и собака Лайка"
                    Твой ответ:
                    питомцы: кот Барсик, собака Лайка
                    """
        
        user_prompt = f"Реплика пользователя: \"{message_content}\""
        
        return system_prompt, user_prompt
    
    def _parse_facts(self, response: str) -> Dict[str, List[str]]:
        """
        Parse the model's response into structured facts dictionary.
        
        Args:
            response: The model's response with facts
            
        Returns:
            Dictionary with facts in format {category: [values]}
        """
        print(f"model extraction-🤖🔍: {response}")
        
        facts: Dict[str, List[str]] = {}
        
        # Parse line by line
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            # Split by first colon
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            
            category = parts[0].strip().lower()
            values_str = parts[1].strip()
            
            # Split values by comma
            values = [v.strip() for v in values_str.split(',') if v.strip()]
            
            if values:
                facts[category] = values
        
        return facts
    
    def summarize_message(self, user_message: str) -> str:
        """
        Summarize new information from user message into concise format.
        Uses local quantized model to extract and compress important information.
        
        Args:
            user_message: The user's message content
            
        Returns:
            Concise summary with only important facts about the user
        """
        system_prompt = """Ты — модуль суммаризации информации о пользователе.
                        Твоя задача — КРАТКО выделить ТОЛЬКО важную информацию о пользователе из его сообщения.

                        КРИТИЧЕСКИ ВАЖНО:
                        - Пиши МАКСИМАЛЬНО КРАТКО, только суть
                        - НЕ добавляй информацию, которой НЕТ в сообщении
                        - НЕ делай предположений или выводов
                        - НЕ добавляй вводные слова и пояснения
                        - Убирай эмоции, вопросы и незначимые детали
                        - Если информация ОБНОВЛЯЕТ старую (переезд, смена работы и т.д.), явно укажи это (например: "теперь", "больше не", "переехал")

                        Примеры:

                        Сообщение: "Я Иван, живу в Москве, работаю программистом"
                        Ответ: Имя Иван, живет в Москве, работает программистом

                        Сообщение: "Классно! Возьму его с собой на футбол"
                        Ответ: играет в футбол

                        Сообщение: "Переехал в Питер, квартиры тут очень дорогие!"
                        Ответ: переехал в Санкт-Петербург

                        Сообщение: "Больше не играю в футбол, теперь баскетболом занимаюсь"
                        Ответ: больше не играет в футбол, теперь играет в баскетбол

                        Сообщение: "Завтра пойду на пробежку утром"
                        Ответ: занимается бегом

                        Сообщение: "У меня есть кот Барсик и собака Лайка"
                        Ответ: есть кот Барсик, собака Лайка

                        Сообщение: "Уволился с работы программистом"
                        Ответ: больше не работает программистом

                        Сообщение: "Как дела?"
                        Ответ: (пусто)"""
        
        user_prompt = f'Сообщение: "{user_message}"'
        
        # Use model for summarization
        prompts = (system_prompt, user_prompt)
        response = self._get_model_response(prompts, max_tokens=100)
        
        return response.strip()


def merge_facts(existing_facts: Dict[str, List[str]], new_facts: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Merge new facts into existing facts dictionary without overwriting.
    
    Args:
        existing_facts: Current facts dictionary
        new_facts: New facts to add
        
    Returns:
        Merged facts dictionary
    """
    merged = existing_facts.copy()
    
    for category, values in new_facts.items():
        if category in merged:
            # Add only new unique values
            existing_values = set(merged[category])
            for value in values:
                if value not in existing_values:
                    merged[category].append(value)
        else:
            # New category
            merged[category] = values.copy()
    
    return merged