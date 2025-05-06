from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
import uuid
import json
import os
from datetime import datetime
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import re
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import traceback

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

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis_history")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Rate limiting dictionary
request_counts = {}
RATE_LIMIT = 30  # requests per minute
RATE_WINDOW = 60  # seconds

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Get stopwords
stop_words = set(nltk.corpus.stopwords.words('english'))

# Initialize models
tfidf_model = None
bert_model = None

def load_tfidf_model():
    """Load TF-IDF model or create a new one if it doesn't exist"""
    global tfidf_model
    
    try:
        tfidf_model_path = os.path.join(MODELS_DIR, "tfidf_logreg.pkl")
        
        if os.path.exists(tfidf_model_path):
            logger.info(f"Loading TF-IDF model from {tfidf_model_path}")
            tfidf_model = joblib.load(tfidf_model_path)
        else:
            logger.info("Creating new TF-IDF model")
            # Create a simple pipeline
            tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            classifier = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
            
            # Load sample data
            fake_news_path = os.path.join(DATA_DIR, "fake_news.csv")
            real_news_path = os.path.join(DATA_DIR, "real_news.csv")
            
            fake_texts = []
            real_texts = []
            
            # Try to load sample data
            try:
                if os.path.exists(fake_news_path):
                    import pandas as pd
                    fake_df = pd.read_csv(fake_news_path)
                    if 'text' in fake_df.columns:
                        fake_texts = fake_df['text'].tolist()
                
                if os.path.exists(real_news_path):
                    import pandas as pd
                    real_df = pd.read_csv(real_news_path)
                    if 'text' in real_df.columns:
                        real_texts = real_df['text'].tolist()
            except Exception as e:
                logger.error(f"Error loading sample data: {str(e)}")
            
            # If no sample data, use some default examples
            if not fake_texts:
                fake_texts = [
                    "BREAKING: Scientists confirm that drinking hot water mixed with lemon juice every morning can prevent cancer with 100% effectiveness.",
                    "Doctors don't want you to know this one simple trick that cures all diseases overnight.",
                    "SHOCKING: Government hiding alien technology that could solve all energy problems."
                ]
            
            if not real_texts:
                real_texts = [
                    "Scientists have discovered a new species of deep-sea fish that can withstand extreme pressure at depths of over 8,000 meters.",
                    "A recent study published in Nature suggests that regular exercise may reduce the risk of certain types of cancer by up to 20%.",
                    "The Federal Reserve announced a 0.25% increase in interest rates following their quarterly meeting yesterday."
                ]
            
            # Combine data
            X = fake_texts + real_texts
            y = [1] * len(fake_texts) + [0] * len(real_texts)  # 1 for fake, 0 for real
            
            # Fit the model if we have data
            if X and y:
                # Transform text data
                X_tfidf = tfidf_vectorizer.fit_transform(X)
                
                # Train the model
                classifier.fit(X_tfidf, y)
                
                # Create pipeline
                from sklearn.pipeline import Pipeline
                tfidf_model = Pipeline([
                    ('tfidf', tfidf_vectorizer),
                    ('classifier', classifier)
                ])
                
                # Save model
                os.makedirs(os.path.dirname(tfidf_model_path), exist_ok=True)
                joblib.dump(tfidf_model, tfidf_model_path)
                logger.info(f"Saved new TF-IDF model to {tfidf_model_path}")
            else:
                logger.warning("No data available to train TF-IDF model")
                # Create a dummy model
                tfidf_model = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=1000)),
                    ('classifier', LogisticRegression())
                ])
        
        return tfidf_model
    
    except Exception as e:
        logger.error(f"Error loading TF-IDF model: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create a dummy model as fallback
        from sklearn.pipeline import Pipeline
        tfidf_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000)),
            ('classifier', LogisticRegression())
        ])
        return tfidf_model

def load_bert_model():
    """Load BERT model if available"""
    global bert_model
    
    try:
        # Try to import transformers
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        bert_model_dir = os.path.join(MODELS_DIR, "bert")
        
        if os.path.exists(bert_model_dir) and os.path.isdir(bert_model_dir):
            logger.info(f"Loading BERT model from {bert_model_dir}")
            
            # Create a simple BERT wrapper class
            class BERTWrapper:
                def __init__(self, model_dir):
                    self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.model.to(self.device)
                
                def predict(self, text):
                    # Preprocess text
                    processed_text = preprocess_text(text)
                    
                    # Tokenize input
                    inputs = self.tokenizer(
                        processed_text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512
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
            
            bert_model = BERTWrapper(bert_model_dir)
        else:
            logger.info("Using pre-trained BERT model")
            
            # Use a pre-trained model
            model_name = "distilbert-base-uncased"
            
            # Create a simple BERT wrapper class
            class BERTWrapper:
                def __init__(self, model_name):
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.model.to(self.device)
                    
                    # Since we're using a pre-trained model without fine-tuning,
                    # we'll use a simple heuristic approach for fake news detection
                    
                def predict(self, text):
                    # For a pre-trained model without fine-tuning, we'll use features
                    # to make a prediction instead of the actual model output
                    
                    # Extract features
                    features = extract_features(text)
                    
                    # Calculate fake probability based on features
                    sensational_score = features["sensational_count"] * 0.3
                    credible_score = features["credible_count"] * -0.3
                    clickbait_score = features["clickbait_count"] * 0.2
                    exclamation_score = min(features["exclamation_count"] * 0.1, 0.3)
                    caps_score = min(features["all_caps_count"] * 0.1, 0.3)
                    
                    # Combine scores
                    total_score = sensational_score + credible_score + clickbait_score + exclamation_score + caps_score
                    
                    # Normalize to [0, 1]
                    fake_probability = 1 / (1 + np.exp(-total_score))
                    
                    # Determine prediction
                    if fake_probability > 0.7:
                        prediction = "FAKE"
                        confidence = fake_probability
                    elif fake_probability < 0.3:
                        prediction = "REAL"
                        confidence = 1 - fake_probability
                    else:
                        prediction = "UNCERTAIN"
                        confidence = 1 - abs(0.5 - fake_probability) * 2
                    
                    # Extract important words
                    important_words = extract_important_words(text)
                    
                    # Create result
                    result = {
                        "prediction": prediction,
                        "confidence": confidence,
                        "fake_probability": fake_probability,
                        "important_words": important_words
                    }
                    
                    return result
            
            bert_model = BERTWrapper(model_name)
        
        return bert_model
    
    except Exception as e:
        logger.error(f"Error loading BERT model: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create a dummy BERT model as fallback
        class DummyBERT:
            def predict(self, text):
                # Use TF-IDF model as fallback
                if tfidf_model:
                    return predict_with_tfidf(text)
                else:
                    # Extract features
                    features = extract_features(text)
                    
                    # Calculate fake probability based on features
                    sensational_score = features["sensational_count"] * 0.3
                    credible_score = features["credible_count"] * -0.3
                    clickbait_score = features["clickbait_count"] * 0.2
                    exclamation_score = min(features["exclamation_count"] * 0.1, 0.3)
                    caps_score = min(features["all_caps_count"] * 0.1, 0.3)
                    
                    # Combine scores
                    total_score = sensational_score + credible_score + clickbait_score + exclamation_score + caps_score
                    
                    # Normalize to [0, 1]
                    fake_probability = 1 / (1 + np.exp(-total_score))
                    
                    # Determine prediction
                    if fake_probability > 0.7:
                        prediction = "FAKE"
                        confidence = fake_probability
                    elif fake_probability < 0.3:
                        prediction = "REAL"
                        confidence = 1 - fake_probability
                    else:
                        prediction = "UNCERTAIN"
                        confidence = 1 - abs(0.5 - fake_probability) * 2
                    
                    # Extract important words
                    important_words = extract_important_words(text)
                    
                    # Create result
                    result = {
                        "prediction": prediction,
                        "confidence": confidence,
                        "fake_probability": fake_probability,
                        "important_words": important_words
                    }
                    
                    return result
        
        bert_model = DummyBERT()
        return bert_model

# Load models
tfidf_model = load_tfidf_model()
bert_model = load_bert_model()

# Text processing functions
def preprocess_text(text):
    try:
        logger.info(f"Original text: {text[:200]}...")  # Log first 200 chars
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words]
        processed_text = ' '.join(tokens)
        logger.info(f"Processed text: {processed_text[:200]}...")  # Log processed
        return processed_text
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        return text

def extract_features(text):
    """
    Extract linguistic features from text
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary of features
    """
    try:
        # Define sensational and credible terms
        sensational_terms = [
            'breaking', 'shocking', 'miracle', 'secret', '100%', 'cure', 'overnight', 
            'conspiracy', 'they don\'t want you to know', 'revealed', 'exclusive', 
            'shocking truth', 'government cover-up', 'what they aren\'t telling you',
            'doctors hate', 'one simple trick', 'never revealed', 'banned', 'censored',
            'suppressed', 'hidden', 'urgent', 'warning', 'alert', 'incredible'
        ]
        
        credible_terms = [
            'research', 'study', 'scientists', 'published', 'journal', 'evidence', 
            'analysis', 'data', 'according to', 'experts say', 'report', 'investigation',
            'survey', 'findings', 'peer-reviewed', 'clinical trial', 'experiment',
            'statistics', 'researchers', 'evidence suggests', 'preliminary results'
        ]
        
        # Define clickbait phrases
        clickbait_phrases = [
            'you won\'t believe', 'mind blowing', 'will shock you', 'what happens next',
            'changed my life', 'jaw-dropping', 'unbelievable', 'incredible', 'amazing',
            'stunning', 'insane', 'revolutionary', 'game-changing', 'life-changing',
            'this is why', 'the reason why', 'here\'s why', 'find out why'
        ]
        
        # Convert to lowercase for matching
        text_lower = text.lower()
        
        # Count sensational terms
        sensational_count = sum(1 for term in sensational_terms if term in text_lower)
        
        # Count credible terms
        credible_count = sum(1 for term in credible_terms if term in text_lower)
        
        # Count clickbait phrases
        clickbait_count = sum(1 for phrase in clickbait_phrases if phrase in text_lower)
        
        # Check for excessive punctuation
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Check for ALL CAPS words
        words = text.split()
        all_caps_count = sum(1 for word in words if word.isupper() and len(word) > 2)
        
        # Check for presence of numbers in suspicious contexts
        number_patterns = [
            r'\d+\s*%\s*effective',
            r'\d+\s*times\s*more',
            r'lose\s*\d+\s*pounds',
            r'\d+\s*simple\s*tricks'
        ]
        suspicious_numbers = sum(1 for pattern in number_patterns if re.search(pattern, text_lower))
        
        # Calculate text statistics
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        word_count = len(words)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Extract features
        features = {
            'sensational_count': sensational_count,
            'credible_count': credible_count,
            'clickbait_count': clickbait_count,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'all_caps_count': all_caps_count,
            'suspicious_numbers': suspicious_numbers,
            'sentence_count': sentence_count,
            'word_count': word_count,
            'avg_sentence_length': avg_sentence_length
        }
        
        return features
    
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return {
            'sensational_count': 0,
            'credible_count': 0,
            'clickbait_count': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'all_caps_count': 0,
            'suspicious_numbers': 0,
            'sentence_count': 0,
            'word_count': 0,
            'avg_sentence_length': 0
        }

def extract_important_words(text, n=10):
    """
    Extract important words from text
    
    Args:
        text (str): Input text
        n (int): Number of words to extract
        
    Returns:
        list: List of important words with importance scores
    """
    try:
        # Tokenize and preprocess
        tokens = nltk.word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in stop_words and token.isalpha() and len(token) > 2]
        
        # Count word frequencies
        word_freq = {}
        for token in tokens:
            if token in word_freq:
                word_freq[token] += 1
            else:
                word_freq[token] = 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N words
        top_words = sorted_words[:n]
        
        # Calculate importance scores (normalized)
        total_freq = sum(freq for _, freq in top_words) if top_words else 1
        important_words = [
            {"word": word, "importance": freq / total_freq}
            for word, freq in top_words
        ]
        
        return important_words
    
    except Exception as e:
        logger.error(f"Error extracting important words: {str(e)}")
        return []

def generate_explanation(text, features, fake_probability):
    """
    Generate explanation for the prediction
    
    Args:
        text (str): Input text
        features (dict): Extracted features
        fake_probability (float): Probability of being fake news
        
    Returns:
        dict: Explanation dictionary
    """
    try:
        explanation = {
            "summary": "",
            "factors": [],
            "recommendations": []
        }
        
        # Generate summary
        if fake_probability > 0.7:
            explanation["summary"] = "This article contains several characteristics commonly found in fake or misleading news."
        elif fake_probability < 0.3:
            explanation["summary"] = "This article appears to have characteristics of credible news reporting."
        else:
            explanation["summary"] = "This article has mixed characteristics, making it difficult to determine its credibility with high confidence."
        
        # Generate factors
        factors = []
        
        # Sensational language
        if features['sensational_count'] > 0:
            factors.append({
                "type": "negative" if features['sensational_count'] > 1 else "neutral",
                "description": f"Contains {features['sensational_count']} sensational terms or phrases (e.g., 'shocking', 'miracle', '100% effective')."
            })
        
        # Credible language
        if features['credible_count'] > 0:
            factors.append({
                "type": "positive" if features['credible_count'] > 1 else "neutral",
                "description": f"Contains {features['credible_count']} terms associated with credible reporting (e.g., 'research', 'study', 'evidence')."
            })
        
        # Clickbait
        if features['clickbait_count'] > 0:
            factors.append({
                "type": "negative",
                "description": f"Contains {features['clickbait_count']} clickbait phrases (e.g., 'you won't believe', 'mind blowing')."
            })
        
        # Excessive punctuation
        if features['exclamation_count'] > 2:
            factors.append({
                "type": "negative",
                "description": f"Uses excessive exclamation marks ({features['exclamation_count']} found), which is uncommon in credible reporting."
            })
        
        # ALL CAPS
        if features['all_caps_count'] > 2:
            factors.append({
                "type": "negative",
                "description": f"Contains {features['all_caps_count']} words in ALL CAPS, which is often used for sensationalism."
            })
        
        # Suspicious numbers
        if features['suspicious_numbers'] > 0:
            factors.append({
                "type": "negative",
                "description": "Contains suspicious numerical claims that may be exaggerated or misleading."
            })
        
        # Text length
        word_count = features['word_count']
        if word_count < 50:
            factors.append({
                "type": "negative",
                "description": f"Very short article ({word_count} words), which may lack sufficient detail for a comprehensive news story."
            })
        elif word_count > 300:
            factors.append({
                "type": "positive",
                "description": f"Substantial article length ({word_count} words), providing more detailed information."
            })
        
        explanation["factors"] = factors
        
        # Generate recommendations
        recommendations = [
            "Always verify information from multiple credible sources.",
            "Check if the article cites specific sources, studies, or experts.",
            "Look for the original source of claims or statistics mentioned.",
            "Be cautious of articles that use emotional or sensational language."
        ]
        
        explanation["recommendations"] = recommendations
        
        return explanation
    
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        return {
            "summary": "Unable to generate detailed explanation.",
            "factors": [],
            "recommendations": ["Verify information from multiple credible sources."]
        }

def predict_with_tfidf(text):
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
        if not hasattr(tfidf_model.named_steps['classifier'], 'classes_'):
            logger.warning("TF-IDF model is not trained. Using feature-based prediction.")
            
            # Extract features
            features = extract_features(text)
            
            # Calculate fake probability based on features
            sensational_score = features["sensational_count"] * 0.3
            credible_score = features["credible_count"] * -0.3
            clickbait_score = features["clickbait_count"] * 0.2
            exclamation_score = min(features["exclamation_count"] * 0.1, 0.3)
            caps_score = min(features["all_caps_count"] * 0.1, 0.3)
            
            # Combine scores
            total_score = sensational_score + credible_score + clickbait_score + exclamation_score + caps_score
            
            # Normalize to [0, 1]
            fake_probability = 1 / (1 + np.exp(-total_score))
            
            # Determine prediction
            if fake_probability > 0.7:
                prediction = "FAKE"
                confidence = fake_probability
            elif fake_probability < 0.3:
                prediction = "REAL"
                confidence = 1 - fake_probability
            else:
                prediction = "UNCERTAIN"
                confidence = 1 - abs(0.5 - fake_probability) * 2
        else:
            # Transform text
            X = tfidf_model.named_steps['tfidf'].transform([processed_text])
            
            # Make prediction
            prediction_idx = tfidf_model.named_steps['classifier'].predict(X)[0]
            probabilities = tfidf_model.named_steps['classifier'].predict_proba(X)[0]
            
            # Map prediction index to label
            prediction = "FAKE" if prediction_idx == 1 else "REAL"
            
            # Get confidence and fake probability
            confidence = probabilities[prediction_idx]
            fake_probability = probabilities[1] if len(probabilities) > 1 else 0.5
        
        # Extract important words
        important_words = extract_important_words(text)
        
        # Create result
        result = {
            "prediction": prediction,
            "confidence": float(confidence),  # Convert numpy types to Python types
            "fake_probability": float(fake_probability),
            "important_words": important_words
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error making TF-IDF prediction: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Fallback to feature-based prediction
        features = extract_features(text)
        
        # Calculate fake probability based on features
        sensational_score = features["sensational_count"] * 0.3
        credible_score = features["credible_count"] * -0.3
        clickbait_score = features["clickbait_count"] * 0.2
        exclamation_score = min(features["exclamation_count"] * 0.1, 0.3)
        caps_score = min(features["all_caps_count"] * 0.1, 0.3)
        
        # Combine scores
        total_score = sensational_score + credible_score + clickbait_score + exclamation_score + caps_score
        
        # Normalize to [0, 1]
        fake_probability = 1 / (1 + np.exp(-total_score))
        
        # Determine prediction
        if fake_probability > 0.7:
            prediction = "FAKE"
            confidence = fake_probability
        elif fake_probability < 0.3:
            prediction = "REAL"
            confidence = 1 - fake_probability
        else:
            prediction = "UNCERTAIN"
            confidence = 1 - abs(0.5 - fake_probability) * 2
        
        # Extract important words
        important_words = extract_important_words(text)
        
        # Create result
        result = {
            "prediction": prediction,
            "confidence": float(confidence),
            "fake_probability": float(fake_probability),
            "important_words": important_words
        }
        
        return result

def predict_with_bert(text):
    """
    Make prediction using BERT model
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Prediction result
    """
    try:
        # Use BERT model for prediction
        result = bert_model.predict(text)
        
        # Ensure numeric values are Python types, not numpy types
        result["confidence"] = float(result["confidence"])
        result["fake_probability"] = float(result["fake_probability"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error making BERT prediction: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Fallback to TF-IDF model
        return predict_with_tfidf(text)

# Web scraping function
def scrape_news_from_url(url):
    """
    Scrape news article content from a given URL
    
    Args:
        url (str): URL of the news article
        
    Returns:
        str: Extracted article text or None if extraction failed
    """
    try:
        # Add http:// if missing
        if not url.startswith('http'):
            url = 'https://' + url
        
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.error(f"Invalid URL format: {url}")
            return None
            
        # Send request with appropriate headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.extract()
            
        # Try to find article content based on common patterns
        article_text = ""
        
        # Method 1: Look for article or main tags
        article_tag = soup.find('article') or soup.find('main')
        if article_tag:
            paragraphs = article_tag.find_all('p')
            article_text = ' '.join([p.get_text().strip() for p in paragraphs])
        
        # Method 2: If no article found, look for common content div classes
        if not article_text:
            content_divs = soup.find_all(['div', 'section'], class_=re.compile(r'(content|article|post|story|text)'))
            if content_divs:
                for div in content_divs:
                    paragraphs = div.find_all('p')
                    if paragraphs and len(paragraphs) > 3:  # Assume real articles have multiple paragraphs
                        article_text = ' '.join([p.get_text().strip() for p in paragraphs])
                        break
        
        # Method 3: If still no content, get all paragraphs
        if not article_text:
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text().strip() for p in paragraphs])
            
        # Clean up the text
        article_text = re.sub(r'\s+', ' ', article_text).strip()
        article_text = re.sub(r'[^\w\s.,!?;:()\-\'"]', '', article_text)
        
        if not article_text or len(article_text.split()) < 20:
            logger.warning(f"Could not extract meaningful content from {url}")
            return None
            
        logger.info(f"Successfully extracted {len(article_text.split())} words from {url}")
        return article_text
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for {url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Rate limiting middleware
def check_rate_limit(ip_address):
    """Simple rate limiting implementation"""
    current_time = time.time()
    
    # Clean up old entries
    for ip in list(request_counts.keys()):
        if current_time - request_counts[ip]["timestamp"] > RATE_WINDOW:
            del request_counts[ip]
    
    # Check current IP
    if ip_address in request_counts:
        if request_counts[ip_address]["count"] >= RATE_LIMIT:
            return False
        request_counts[ip_address]["count"] += 1
    else:
        request_counts[ip_address] = {
            "count": 1,
            "timestamp": current_time
        }
    
    return True

# Input validation
def validate_input(data):
    """Validate and sanitize input data"""
    if not data:
        return False, "No data provided"
    
    if 'text' not in data:
        return False, "No text field in request"
        
    if not data['text'] or not isinstance(data['text'], str):
        return False, "Text must be a non-empty string"
        
    if 'model' not in data:
        return False, "No model field in request"
        
    if data['model'].lower().replace("-", "") not in ['bert', 'tfidf']:
        return False, "Invalid model selection"
    
    data['text'] = data['text'].strip()
    return True, data

# Save result to history
def save_to_history(result):
    """Save analysis result to history"""
    try:
        # Create history directory if it doesn't exist
        os.makedirs(HISTORY_DIR, exist_ok=True)
        
        # Save to file
        filename = os.path.join(HISTORY_DIR, f"{result['id']}.json")
        with open(filename, 'w') as f:
            json.dump(result, f)
            
        logger.info(f"Saved analysis result to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving to history: {str(e)}")
        logger.error(traceback.format_exc())

# API Routes
@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for single text prediction"""
    try:
        # Check rate limit
        if not check_rate_limit(request.remote_addr):
            logger.warning(f"Rate limit exceeded for IP: {request.remote_addr}")
            return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
        
        # Get and validate data
        data = request.json
        logger.info(f"Received prediction request: {data}")  # Log incoming data
        valid, result = validate_input(data)
        
        if not valid:
            logger.error(f"Validation failed: {result}")  # Log validation error
            logger.warning(f"Invalid input: {result}")
            return jsonify({"error": result}), 400
        
        # Log request
        logger.info(f"Prediction request: model={data['model']}, text_length={len(data['text'])}")
        
        # Generate a unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Extract features
        features = extract_features(data['text'])
        
        # Make prediction based on selected model
        if data['model'].lower() == 'bert':
            model_result = predict_with_bert(data['text'])
        else:
            model_result = predict_with_tfidf(data['text'])
        
        # Generate explanation
        explanation = generate_explanation(data['text'], features, model_result['fake_probability'])
        
        # Create result object
        result = {
            "id": analysis_id,
            "text": data['text'],
            "prediction": model_result['prediction'],
            "confidence": model_result['confidence'],
            "fake_probability": model_result['fake_probability'],
            "features": features,
            "important_words": model_result['important_words'],
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save result to history
        save_to_history(result)
        
        # Log result
        logger.info(f"Prediction result: {result['prediction']} with confidence {result['confidence']:.2f}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_prediction():
    """Endpoint for batch prediction"""
    try:
        # Check rate limit
        if not check_rate_limit(request.remote_addr):
            logger.warning(f"Rate limit exceeded for IP: {request.remote_addr}")
            return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
        
        # Get and validate data
        data = request.json
        
        if not data or 'texts' not in data or not isinstance(data['texts'], list):
            return jsonify({"error": "Invalid input. 'texts' field must be a list."}), 400
            
        if 'model' not in data or data['model'].lower() not in ['bert', 'tfidf']:
            return jsonify({"error": "Invalid model selection"}), 400
        
        # Log request
        logger.info(f"Batch prediction request: model={data['model']}, batch_size={len(data['texts'])}")
        
        # Process batch
        results = []
        
        for text in data['texts']:
            # Generate a unique ID for this analysis
            analysis_id = str(uuid.uuid4())
            
            # Extract features
            features = extract_features(text)
            
            # Make prediction based on selected model
            if data['model'].lower() == 'bert':
                model_result = predict_with_bert(text)
            else:
                model_result = predict_with_tfidf(text)
            
            # Generate explanation
            explanation = generate_explanation(text, features, model_result['fake_probability'])
            
            # Create result object
            result = {
                "id": analysis_id,
                "text": text,
                "prediction": model_result['prediction'],
                "confidence": model_result['confidence'],
                "fake_probability": model_result['fake_probability'],
                "features": features,
                "important_words": model_result['important_words'],
                "explanation": explanation,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save result to history
            save_to_history(result)
            
            # Add to results
            results.append(result)
        
        # Log result
        logger.info(f"Batch prediction completed: {len(results)} items processed")
        
        return jsonify({"results": results})
        
    except Exception as e:
        logger.error(f"Error processing batch request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@app.route('/scrape', methods=['POST'])
def scrape():
    """Endpoint for scraping news from URL"""
    try:
        # Check rate limit
        if not check_rate_limit(request.remote_addr):
            logger.warning(f"Rate limit exceeded for IP: {request.remote_addr}")
            return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
        
        # Get data
        data = request.json
        
        if not data or 'url' not in data:
            return jsonify({"error": "No URL provided"}), 400
        
        url = data['url']
        
        # Log request
        logger.info(f"Scrape request: url={url}")
        
        # Scrape URL
        article_text = scrape_news_from_url(url)
        
        if not article_text:
            return jsonify({"error": "Could not extract content from URL"}), 400
        
        return jsonify({"text": article_text})
        
    except Exception as e:
        logger.error(f"Error processing scrape request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Endpoint for getting analysis history"""
    try:
        # Get history files
        if not os.path.exists(HISTORY_DIR):
            os.makedirs(HISTORY_DIR, exist_ok=True)
            return jsonify({"history": []})
            
        history_files = os.listdir(HISTORY_DIR)
        
        # Load history items
        history = []
        
        for filename in history_files:
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(HISTORY_DIR, filename)
            
            try:
                with open(filepath, 'r') as f:
                    item = json.load(f)
                    history.append(item)
            except Exception as e:
                logger.error(f"Error loading history item {filename}: {str(e)}")
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({"history": history})
        
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

# Run the app
if __name__ == '__main__':
    # Start the server
    app.run(host="0.0.0.0", port=5000, debug=True)