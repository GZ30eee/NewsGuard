import joblib
import os
import logging
from sklearn.pipeline import Pipeline
import config

logger = logging.getLogger(__name__)

class TFIDFModel:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    def _load(self):
        """Lazy loader for TF-IDF pipeline"""
        try:
            if os.path.exists(config.TFIDF_MODEL_PATH):
                logger.info("Loading TF-IDF model...")
                self._model = joblib.load(config.TFIDF_MODEL_PATH)
            else:
                logger.warning("TF-IDF model not found. Initializing empty pipeline.")
                self._model = self._create_empty_pipeline()
        except Exception as e:
            logger.error(f"TF-IDF loading failed: {str(e)}")
            self._model = self._create_empty_pipeline()

    def _create_empty_pipeline(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        return Pipeline([
            ('tfidf', TfidfVectorizer(max_features=config.MAX_FEATURES)),
            ('classifier', LogisticRegression())
        ])

    def predict(self, text):
        """Prediction logic moved partially to PredictionService for better orchestration"""
        # This wrapper remains for direct access if needed, 
        # but the PredictionService handles the bulk now for consistency.
        pass

    def save_model(self, path=None):
        save_path = path or config.TFIDF_MODEL_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.model, save_path)
        logger.info(f"Model saved to {save_path}")
