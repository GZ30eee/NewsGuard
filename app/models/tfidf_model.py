import joblib
import numpy as np
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import config
from app.utils.text_processor import preprocess_text, extract_important_words

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TFIDFModel:
    def __init__(self):
        # Load or create model
        self.model = self._load_model()
        
    def _load_model(self):
        """
        Load TF-IDF model from disk or create a new one
        
        Returns:
            sklearn.pipeline.Pipeline: Model pipeline
        """
        try:
            # Check if model exists
            if os.path.exists(config.TFIDF_MODEL_PATH):
                logger.info(f"Loading TF-IDF model from {config.TFIDF_MODEL_PATH}")
                return joblib.load(config.TFIDF_MODEL_PATH)
            else:
                logger.info("Creating new TF-IDF model")
                # Create new model
                pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=config.MAX_FEATURES, ngram_range=config.NGRAM_RANGE)),
                    ('classifier', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42))
                ])
                return pipeline
                
        except Exception as e:
            logger.error(f"Error loading TF-IDF model: {str(e)}")
            # Create a simple fallback model
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000)),
                ('classifier', LogisticRegression())
            ])
            return pipeline
    
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
            
            # Check if model is trained
            if not hasattr(self.model.named_steps['classifier'], 'classes_'):
                logger.warning("Model is not trained. Returning default prediction.")
                return {
                    "prediction": "UNCERTAIN",
                    "confidence": 0.5,
                    "fake_probability": 0.5,
                    "important_words": []
                }
            
            # Transform text
            X = self.model.named_steps['tfidf'].transform([processed_text])
            
            # Make prediction
            prediction = self.model.named_steps['classifier'].predict(X)[0]
            probabilities = self.model.named_steps['classifier'].predict_proba(X)[0]
            confidence = probabilities[prediction]
            
            # Extract important words
            important_words = extract_important_words(text)
            
            # Create result
            result = {
                "prediction": "FAKE" if prediction == 1 else "REAL",
                "confidence": confidence,
                "fake_probability": probabilities[1],
                "important_words": important_words
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making TF-IDF prediction: {str(e)}")
            return {
                "prediction": "UNCERTAIN",
                "confidence": 0.5,
                "fake_probability": 0.5,
                "important_words": []
            }
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train (list): List of training texts
            y_train (list): List of training labels (0 for real, 1 for fake)
            
        Returns:
            float: Accuracy score
        """
        try:
            logger.info("Training TF-IDF model")
            
            # Fit model
            self.model.fit(X_train, y_train)
            
            # Save model
            self.save_model(config.TFIDF_MODEL_PATH)
            
            # Return accuracy on training data
            return self.model.score(X_train, y_train)
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return 0.0
    
    def save_model(self, path):
        """
        Save model to disk
        
        Args:
            path (str): Path to save model
        """
        try:
            logger.info(f"Saving model to {path}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise