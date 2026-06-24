import logging
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class ImportanceClassifier:
    def __init__(self, model_path: str, threshold: float = 0.5):
        self.threshold = threshold
        self._stub = not model_path
        self.tokenizer: Any = None
        self.model: Any = None
        self.device = "cpu"

        if self._stub:
            logger.info(
                "ImportanceClassifier stub mode (no model_path) threshold=%.3f",
                threshold,
            )
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Initializing ImportanceClassifier model_path=%s threshold=%.3f device=%s",
            model_path,
            threshold,
            self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict[str, float]:
        if self._stub:
            return {
                "p_not_important": 0.1,
                "p_important": 0.9,
                "is_important": True,
            }

        import torch

        logger.debug("Classifier predict text_len=%d", len(text))
        # Use tokenizer.encode() instead of tokenizer.__call__() to avoid
        # a Python 3.13 / tokenizers compatibility issue where encode_batch
        # raises TypeError on valid strings.
        token_ids = self.tokenizer.encode(
            "query: " + text,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
        )
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
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
