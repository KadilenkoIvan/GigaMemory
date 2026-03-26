from typing import Dict
import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class ImportanceClassifier:
    def __init__(self, model_path: str, threshold: float = 0.5):
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Initializing ImportanceClassifier model_path=%s threshold=%.3f device=%s",
            model_path,
            threshold,
            self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Dict[str, float]:
        logger.debug("Classifier predict text_len=%d", len(text))
        inputs = self.tokenizer(
            "query: " + text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        p_not_important = float(probs[0].item())
        p_important = float(probs[1].item())
        return {
            "p_not_important": p_not_important,
            "p_important": p_important,
            "is_important": p_important >= self.threshold,
        }
