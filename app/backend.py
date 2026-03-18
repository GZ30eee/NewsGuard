from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
import os
import config
from app.routes.news_routes import news_bp
from app.utils.logging_utils import setup_logging
from app.services.prediction_service import PredictionService
import joblib

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup structured logging
setup_logging(app)
logger = logging.getLogger(__name__)

# Rate Limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[f"{config.RATE_LIMIT} per minute"],
    storage_uri="memory://"
)

def get_tfidf_model():
    """Lazy loader for TF-IDF model"""
    if not hasattr(app, '_tfidf_model'):
        try:
            if os.path.exists(config.TFIDF_MODEL_PATH):
                logger.info("Loading TF-IDF model...")
                app._tfidf_model = joblib.load(config.TFIDF_MODEL_PATH)
            else:
                logger.warning("TF-IDF model path not found. Running training placeholder...")
                # Training logic moved to train_models.py, but for quick dev fallback
                from sklearn.pipeline import Pipeline
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression
                app._tfidf_model = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=config.MAX_FEATURES)),
                    ('classifier', LogisticRegression())
                ])
        except Exception as e:
            logger.error(f"Failed to load TF-IDF model: {str(e)}")
            raise
    return app._tfidf_model

def get_bert_model():
    """Lazy loader for BERT model"""
    if not hasattr(app, '_bert_model'):
        try:
            # We'll use the implementations in app/models if they exist
            # For now, let's look for the wrapper
            from app.models.bert_model import BERTModel
            logger.info("Initializing BERT model...")
            app._bert_model = BERTModel()
        except Exception as e:
            logger.warning(f"Failed to load BERT model: {str(e)}. Fallback to None.")
            app._bert_model = None
    return app._bert_model

@app.before_request
def initialize_services():
    """Ensure services are initialized before first request"""
    if not hasattr(app, 'prediction_service'):
        tfidf = get_tfidf_model()
        bert = get_bert_model()
        app.prediction_service = PredictionService(tfidf, bert)

# Register Blueprints
app.register_blueprint(news_bp, url_prefix='/api')

@app.route('/health')
def health_check():
    return {"status": "healthy", "version": config.VERSION}

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.HISTORY_DIR, exist_ok=True)
    
    logger.info(f"Starting {config.APP_NAME} Backend on port {config.API_PORT}")
    app.run(host=config.API_HOST, port=config.API_PORT, debug=config.DEBUG)