import logging
import numpy as np
import config
from app.utils.text_processor import extract_features, generate_explanation, extract_important_words

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, tfidf_model, bert_model):
        self.tfidf_model = tfidf_model
        self.bert_model = bert_model

    def predict(self, text, model_type='tfidf'):
        """Orchestrate prediction using specified model"""
        try:
            if model_type == 'bert' and self.bert_model:
                result = self.bert_model.predict(text)
            else:
                result = self._predict_with_tfidf(text)
            
            # Combine with features and explanation
            features = extract_features(text)
            explanation = generate_explanation(text, features, result['fake_probability'])
            
            # Final result assembly
            result.update({
                "features": features,
                "explanation": explanation,
                "text": text,
                "model_used": model_type
            })
            
            return result
        except Exception as e:
            logger.error(f"Prediction service error: {str(e)}")
            raise

    def ensemble_predict(self, text):
        """Combine BERT and TF-IDF predictions"""
        try:
            tfidf_res = self._predict_with_tfidf(text)
            
            # Try BERT if available
            try:
                bert_res = self.bert_model.predict(text)
                # Weighted average (60% BERT, 40% TF-IDF)
                fake_prob = (bert_res['fake_probability'] * 0.6) + (tfidf_res['fake_probability'] * 0.4)
                model_used = 'ensemble'
            except Exception as e:
                logger.warning(f"BERT ensemble fallback to TF-IDF: {str(e)}")
                fake_prob = tfidf_res['fake_probability']
                model_used = 'tfidf'

            # Determine label
            if fake_prob > config.FAKE_THRESHOLD:
                prediction = "FAKE"
                confidence = fake_prob
            elif fake_prob < config.REAL_THRESHOLD:
                prediction = "REAL"
                confidence = 1 - fake_prob
            else:
                prediction = "UNCERTAIN"
                confidence = 1 - abs(0.5 - fake_prob) * 2

            features = extract_features(text)
            explanation = generate_explanation(text, features, fake_prob)
            
            return {
                "prediction": prediction,
                "confidence": float(confidence),
                "fake_probability": float(fake_prob),
                "important_words": tfidf_res['important_words'],
                "features": features,
                "explanation": explanation,
                "text": text,
                "model_used": model_used
            }
        except Exception as e:
            logger.error(f"Ensemble prediction error: {str(e)}")
            raise

    def _predict_with_tfidf(self, text):
        """Internal TF-IDF prediction logic"""
        from app.utils.text_processor import preprocess_text
        processed_text = preprocess_text(text)
        
        # Check if model is trained
        if not hasattr(self.tfidf_model.named_steps['classifier'], 'classes_'):
            # Fallback to feature-based probability if not trained
            features = extract_features(text)
            fake_prob = self._heuristic_prob(features)
        else:
            X = self.tfidf_model.named_steps['tfidf'].transform([processed_text])
            prediction_idx = self.tfidf_model.named_steps['classifier'].predict(X)[0]
            probabilities = self.tfidf_model.named_steps['classifier'].predict_proba(X)[0]
            fake_prob = probabilities[1] if len(probabilities) > 1 else 0.5

        # Determine label
        if fake_prob > config.FAKE_THRESHOLD:
            prediction = "FAKE"
            confidence = fake_prob
        elif fake_prob < config.REAL_THRESHOLD:
            prediction = "REAL"
            confidence = 1 - fake_prob
        else:
            prediction = "UNCERTAIN"
            confidence = 1 - abs(0.5 - fake_prob) * 2
            
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "fake_probability": float(fake_prob),
            "important_words": extract_important_words(text)
        }

    def _heuristic_prob(self, features):
        """Calculate a fallback probability based on linguistic features"""
        sensational_score = features["sensational_count"] * 0.3
        credible_score = features["credible_count"] * -0.3
        clickbait_score = features["clickbait_count"] * 0.2
        exclamation_score = min(features["exclamation_count"] * 0.1, 0.3)
        caps_score = min(features["all_caps_count"] * 0.1, 0.3)
        
        total_score = sensational_score + credible_score + clickbait_score + exclamation_score + caps_score
        return 1 / (1 + np.exp(-total_score))
