import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import logging
import os
from functools import lru_cache
import config
from app.utils.text_processor import preprocess_text, extract_important_words

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTModel:
    def __init__(self):
        self.model_name = config.BERT_MODEL_NAME
        self.max_length = config.MAX_LENGTH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer, self.model = self._load_model()
        
    @lru_cache(maxsize=1)
    def _load_model(self):
        """
        Load BERT model and tokenizer
        
        Returns:
            tuple: (tokenizer, model)
        """
        try:
            logger.info(f"Loading BERT model: {self.model_name}")
            
            # Check if fine-tuned model exists
            if os.path.exists(config.BERT_MODEL_DIR):
                logger.info(f"Loading fine-tuned model from {config.BERT_MODEL_DIR}")
                tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_DIR)
                model = AutoModelForSequenceClassification.from_pretrained(config.BERT_MODEL_DIR)
            else:
                # Load pre-trained model
                logger.info(f"Loading pre-trained model: {self.model_name}")
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
            
            # Move model to device
            model.to(self.device)
            
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"Error loading BERT model: {str(e)}")
            raise
    
    def predict(self, text):
        """
        Make prediction using BERT model
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Prediction result
        """
        try:
            # Preprocess text
            processed_text = preprocess_text(text)
            
            # Tokenize input
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            # Extract important words
            important_words = extract_important_words(text)
            
            # Create result
            result = {
                "prediction": "FAKE" if prediction == 1 else "REAL",
                "confidence": confidence,
                "fake_probability": probabilities[0][1].item(),
                "important_words": important_words
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making BERT prediction: {str(e)}")
            return {
                "prediction": "UNCERTAIN",
                "confidence": 0.5,
                "fake_probability": 0.5,
                "important_words": []
            }
    
    def save_model(self, path):
        """
        Save model to disk
        
        Args:
            path (str): Path to save model
        """
        try:
            logger.info(f"Saving model to {path}")
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise