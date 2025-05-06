import numpy as np
import logging
import traceback
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from app.utils.text_processor import preprocess_text, extract_features, extract_important_words

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TFIDFModel:
    def __init__(self, model_path="models/tfidf_logreg.pkl"):
        try:
            self.model = joblib.load(model_path)
            logger.info("TF-IDF model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load TF-IDF model: {str(e)}. Initializing untrained model.")
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('classifier', LogisticRegression())
            ])

    def predict(self, text):
        """
        Make prediction using TF-IDF model
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Prediction result
        """
        try:
            # Preprocess text
            processed_text = preprocess_text(text)
            if not processed_text:
                logger.warning("Empty text after preprocessing")
                return self._fallback_prediction(text)

            # Check if model is trained
            if not hasattr(self.model.named_steps['classifier'], 'classes_'):
                logger.warning("TF-IDF model is not trained. Using feature-based prediction.")
                return self._fallback_prediction(text)

            # Transform text
            X = self.model.named_steps['tfidf'].transform([processed_text])

            # Make prediction
            prediction_idx = self.model.named_steps['classifier'].predict(X)[0]
            probabilities = self.model.named_steps['classifier'].predict_proba(X)[0]

            # Map prediction index to label
            prediction = "FAKE" if prediction_idx == 1 else "REAL"

            # Get confidence and fake probability
            confidence = probabilities[prediction_idx]
            fake_probability = probabilities[1] if len(probabilities) > 1 else 0.5

            # Extract important words
            important_words = extract_important_words(text)

            result = {
                "prediction": prediction,
                "confidence": float(confidence),
                "fake_probability": float(fake_probability),
                "important_words": important_words
            }

            logger.info(f"TF-IDF prediction: {prediction}, confidence: {confidence:.2f}")
            return result

        except Exception as e:
            logger.error(f"Error making TF-IDF prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return self._fallback_prediction(text)

    def _fallback_prediction(self, text):
        """Fallback prediction based on features."""
        try:
            features = extract_features(text)
            sensational_score = features["sensational_count"] * 0.3
            credible_score = features["credible_count"] * -0.3
            clickbait_score = features["clickbait_count"] * 0.2
            exclamation_score = min(features["exclamation_count"] * 0.1, 0.3)
            caps_score = min(features["all_caps_count"] * 0.1, 0.3)

            total_score = sensational_score + credible_score + clickbait_score + exclamation_score + caps_score
            fake_probability = 1 / (1 + np.exp(-total_score))

            if fake_probability > 0.7:
                prediction = "FAKE"
                confidence = fake_probability
            elif fake_probability < 0.3:
                prediction = "REAL"
                confidence = 1 - fake_probability
            else:
                prediction = "UNCERTAIN"
                confidence = 1 - abs(0.5 - fake_probability) * 2

            important_words = extract_important_words(text)

            result = {
                "prediction": prediction,
                "confidence": float(confidence),
                "fake_probability": float(fake_probability),
                "important_words": important_words
            }

            logger.info(f"Fallback prediction: {prediction}, confidence: {confidence:.2f}")
            return result

        except Exception as e:
            logger.error(f"Error in fallback prediction: {str(e)}")
            return {
                "prediction": "UNCERTAIN",
                "confidence": 0.5,
                "fake_probability": 0.5,
                "important_words": []
            }

class BERTModel:
    def __init__(self, model_path="models/bert"):
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            logger.info("BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {str(e)}")
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
            if not processed_text:
                logger.warning("Empty text after preprocessing")
                return self._fallback_prediction(text)

            # Tokenize and encode
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

            # Get prediction and confidence
            prediction_idx = np.argmax(probabilities)
            prediction = "FAKE" if prediction_idx == 1 else "REAL"
            confidence = probabilities[prediction_idx]
            fake_probability = probabilities[1] if len(probabilities) > 1 else 0.5

            # Extract important words
            important_words = extract_important_words(text)

            result = {
                "prediction": prediction,
                "confidence": float(confidence),
                "fake_probability": float(fake_probability),
                "important_words": important_words
            }

            logger.info(f"BERT prediction: {prediction}, confidence: {confidence:.2f}")
            return result

        except Exception as e:
            logger.error(f"Error making BERT prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return self._fallback_prediction(text)

    def _fallback_prediction(self, text):
        """Fallback to TF-IDF-style prediction."""
        return TFIDFModel().predict(text)