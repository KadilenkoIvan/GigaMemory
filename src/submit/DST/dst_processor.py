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
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели DST {self.model_path}: {str(e)}")
    
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