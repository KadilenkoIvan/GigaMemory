import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from models import Message


class DSTProcessor:
    """
    Dialog State Tracking processor that determines if a user message should be saved to memory.
    Uses Qwen3-1.7B model to make decisions.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the DST processor with a model path.
        
        Args:
            model_path: Path to the model. If None, will try to use default path.
        """
        if model_path is None:
            # Try to find model in default location
            base_dir = Path(__file__).resolve().parent
            model_path = base_dir / "../models" / "Qwen3-1.7B"
        
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
    
    def should_save_message(self, message: Message) -> Tuple[bool, str]:
        """
        Determine if a message should be saved to memory.
        
        Args:
            message: The message to evaluate
            
        Returns:
            Tuple of (should_save, reason)
            - should_save: True if message should be saved, False otherwise
            - reason: Explanation of the decision
        """
        
        # Construct system and user prompts for the model
        prompts = self._construct_prompt(message.content)
        
        # Get model prediction using the new format
        response = self._get_model_response(prompts)
        
        # Parse the response
        should_save, reason = self._parse_response(response)
        
        return should_save, reason
    
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
                    - "true" — если реплика содержит информацию о пользователе: его интересы, хобби, род занятий, спортивные увлечения, место проживания, возраст, упоминания родственников и друзей, имущество пользователся, транспорт, привычки, вредные привычки, намерения, планы, факты о себе, возможные будущие действия. Даже если это выражено неявно, например: "О, это хороший совет, попробую взять его с собой на футбол." это важно, т.к. пользователь занимается футболом. 
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
            
        Returns:
            The model's response as a string
        """
        system_prompt, user_prompt = prompts
        
        # Формируем сообщения в формате чата
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
        #inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,  # Short response expected
                do_sample=False,    # Greedy decoding
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
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