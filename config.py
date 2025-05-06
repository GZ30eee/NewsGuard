import os

# Application settings
APP_NAME = "Fake News Detector"
VERSION = "1.0.0"
DEBUG = True

# API settings
API_HOST = "0.0.0.0"
API_PORT = 5000
API_URL = f"http://localhost:{API_PORT}"

# Streamlit settings
STREAMLIT_PORT = 8501

# Model settings
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
BERT_MODEL_DIR = os.path.join(MODELS_DIR, "bert")
TFIDF_MODEL_PATH = os.path.join(MODELS_DIR, "tfidf_logreg.pkl")

# Data settings
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FAKE_NEWS_PATH = os.path.join(DATA_DIR, "fake_news.csv")
REAL_NEWS_PATH = os.path.join(DATA_DIR, "real_news.csv")

# History settings
HISTORY_DIR = os.path.join(os.path.dirname(__file__), "analysis_history")
os.makedirs(HISTORY_DIR, exist_ok=True)

# BERT model settings
BERT_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512

# TF-IDF model settings
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)

# Web scraping settings
REQUEST_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Rate limiting
RATE_LIMIT = 10  # requests per minute
RATE_WINDOW = 60  # seconds