import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import os
import config
from functools import lru_cache

logger = logging.getLogger(__name__)

class BERTModel:
    def __init__(self):
        self.model_name = config.BERT_MODEL_NAME
        self.max_length = config.MAX_LENGTH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    def _load(self):
        """Lazy loader for model and tokenizer"""
        try:
            logger.info(f"Loading BERT model on {self.device}...")
            # Check for local fine-tuned model first
            if os.path.exists(config.BERT_MODEL_DIR) and any(os.scandir(config.BERT_MODEL_DIR)):
                path = config.BERT_MODEL_DIR
            else:
                path = self.model_name
            
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2)
            self._model.to(self.device).eval()
            logger.info("BERT model loaded successfully.")
        except Exception as e:
            logger.error(f"BERT loading failed: {str(e)}")
            raise

    def predict(self, text):
        """
        Inference with optimization checks and caching
        """
        try:
            # Basic tokenization
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_idx].item()
                fake_prob = probs[0][1].item() # Assuming label 1 is FAKE

            return {
                "prediction": "FAKE" if pred_idx == 1 else "REAL",
                "confidence": float(confidence),
                "fake_probability": float(fake_prob),
                "important_words": [] # Service will fill this
            }
        except Exception as e:
            logger.error(f"BERT prediction failed: {str(e)}")
            # Graceful fallback handled by PredictionService
            raise