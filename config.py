import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Application settings
APP_NAME = os.getenv("APP_NAME", "NewsGuard")
VERSION = "1.2.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-it")

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 5000))
API_URL = os.getenv("API_URL", f"http://localhost:{API_PORT}")

# Streamlit settings
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model settings
MODELS_DIR = os.path.join(BASE_DIR, "models")
BERT_MODEL_DIR = os.path.join(MODELS_DIR, "bert")
TFIDF_MODEL_PATH = os.path.join(MODELS_DIR, "tfidf_logreg.pkl")

# Data settings
DATA_DIR = os.path.join(BASE_DIR, "data")
FAKE_NEWS_PATH = os.path.join(DATA_DIR, "fake_news.csv")
REAL_NEWS_PATH = os.path.join(DATA_DIR, "real_news.csv")

# History settings
HISTORY_DIR = os.path.join(BASE_DIR, "analysis_history")
DATABASE_URI = os.getenv("DATABASE_URI", f"sqlite:///{os.path.join(BASE_DIR, 'newsguard.db')}")

# BERT model settings
BERT_MODEL_NAME = os.getenv("BERT_MODEL_NAME", "distilbert-base-uncased")
MAX_LENGTH = 512

# TF-IDF model settings
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)

# Web scraping settings
REQUEST_TIMEOUT = 20  # Increased for slow sites
USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

# Rate limiting
RATE_LIMIT = int(os.getenv("RATE_LIMIT", 60))  # Relaxed slightly
RATE_WINDOW = 60  # seconds

# Model thresholds
FAKE_THRESHOLD = 0.7
REAL_THRESHOLD = 0.3

# Ensure directories exist
for directory in [MODELS_DIR, DATA_DIR, HISTORY_DIR]:
    os.makedirs(directory, exist_ok=True)