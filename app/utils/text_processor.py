import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import logging
import config

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

# Download required NLTK data with error handling
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK data (punkt, punkt_tab, stopwords) downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {str(e)}")
        raise

download_nltk_data()

# Use config feature lists if available, else fallback
SENSATIONAL_WORDS = getattr(config, 'SENSATIONAL_WORDS', [
    'breaking', 'shocking', 'miracle', 'secret', '100%', 'cure', 'overnight',
    'conspiracy', 'they don\'t want you to know', 'revealed', 'exclusive',
    'shocking truth', 'government cover-up', 'what they aren\'t telling you',
    'doctors hate', 'one simple trick', 'never revealed', 'banned', 'censored',
    'suppressed', 'hidden', 'urgent', 'warning', 'alert', 'incredible',
    'devastating', 'terrifying', 'scandalous', 'game changer', 'mysterious',
    'baffled', 'unidentified', 'alien', 'unprecedented', 'strange', 'unknown',
    'anomaly', 'phenomenon'
])
CREDIBLE_TERMS = getattr(config, 'CREDIBLE_TERMS', [
    'research', 'study', 'scientists', 'published', 'journal', 'evidence',
    'analysis', 'data', 'according to', 'experts say', 'report', 'investigation',
    'survey', 'findings', 'peer-reviewed', 'clinical trial', 'experiment',
    'statistics', 'researchers', 'evidence suggests', 'preliminary results',
    'university', 'expert', 'institute', 'organisation'
])
CLICKBAIT_PHRASES = getattr(config, 'CLICKBAIT_PHRASES', [
    'you won\'t believe', 'mind blowing', 'will shock you', 'what happens next',
    'changed my life', 'jaw-dropping', 'unbelievable', 'incredible', 'amazing',
    'stunning', 'insane', 'revolutionary', 'game-changing', 'life-changing',
    'this is why', 'the reason why', 'here\'s why', 'find out why',
    'you won’t believe', 'game changer', 'never seen before', 'global phenomenon',
    'too early to speculate', 'wildly varied'
])

def preprocess_text(text):
    """Preprocess text for feature extraction."""
    try:
        if not isinstance(text, str) or not text.strip():
            logger.warning("Empty or invalid input text")
            return ""
        # Preserve case for ALL CAPS detection, normalize spaces
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        return ""

def extract_features(text):
    """
    Extract linguistic features from text
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary of features
    """
    try:
        # Initialize default features
        features = {
            'sensational_count': 0,
            'credible_count': 0,
            'clickbait_count': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'all_caps_count': 0,
            'suspicious_numbers': 0,
            'speculative_count': 0,
            'sentence_count': 0,
            'word_count': 0,
            'avg_sentence_length': 0
        }

        # Preprocess text
        processed_text = preprocess_text(text)
        if not processed_text:
            logger.warning("No text after preprocessing")
            return features

        # Convert to lowercase for term matching
        text_lower = processed_text.lower()

        # Tokenize sentences and words with fallback
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.warning(f"Sent_tokenize failed: {str(e)}. Falling back to basic splitting.")
            sentences = text.split('.')

        try:
            tokens = word_tokenize(text_lower)
        except Exception as e:
            logger.warning(f"Word_tokenize failed: {str(e)}. Falling back to basic splitting.")
            tokens = text_lower.split()

        words = [t for t in tokens if t.isalnum()]

        # Update sentence and word counts
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        features['word_count'] = len(words)
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0

        # Count sensational terms
        matched_sensational = []
        for term in SENSATIONAL_WORDS:
            count = len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
            features['sensational_count'] += count
            if count > 0:
                matched_sensational.append(f"{term}: {count}")
        if matched_sensational:
            logger.info(f"Matched sensational terms: {', '.join(matched_sensational)}")

        # Count credible terms
        matched_credible = []
        for term in CREDIBLE_TERMS:
            count = len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
            features['credible_count'] += count
            if count > 0:
                matched_credible.append(f"{term}: {count}")
        if matched_credible:
            logger.info(f"Matched credible terms: {', '.join(matched_credible)}")

        # Count clickbait phrases
        matched_clickbait = []
        for phrase in CLICKBAIT_PHRASES:
            count = len(re.findall(r'\b' + re.escape(phrase) + r'\b', text_lower))
            features['clickbait_count'] += count
            if count > 0:
                matched_clickbait.append(f"{phrase}: {count}")
        if matched_clickbait:
            logger.info(f"Matched clickbait phrases: {', '.join(matched_clickbait)}")

        # Count speculative language
        speculative_phrases = ['could be', 'may be', 'possibly', 'speculation', 'suggests that']
        for phrase in speculative_phrases:
            features['speculative_count'] += len(re.findall(r'\b' + re.escape(phrase) + r'\b', text_lower))

        # Count punctuation
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')

        # Count ALL CAPS words
        original_words = text.split()
        features['all_caps_count'] = sum(1 for word in original_words if word.isupper() and len(word) > 2 and word.isalpha())

        # Check for suspicious numbers
        number_patterns = [
            r'\d+\s*%\s*effective',
            r'\d+\s*times\s*more',
            r'lose\s*\d+\s*pounds',
            r'\d+\s*simple\s*tricks',
            r'\bmillions?\b',
            r'\bbillions?\b',
            r'\btrillions?\b'
        ]
        for pattern in number_patterns:
            features['suspicious_numbers'] += len(re.findall(pattern, text_lower))

        logger.info(f"Extracted features: {features}")
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
            'speculative_count': 0,
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
        # Preprocess and tokenize
        processed_text = preprocess_text(text)
        if not processed_text:
            logger.warning("No text for important words extraction")
            return []

        try:
            tokens = word_tokenize(processed_text.lower())
        except Exception as e:
            logger.warning(f"Word_tokenize failed: {str(e)}. Falling back to basic splitting.")
            tokens = processed_text.lower().split()

        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2]

        if not tokens:
            logger.warning("No valid tokens after filtering")
            return []

        # Count word frequencies
        word_freq = {}
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:n]

        # Calculate importance scores
        total_freq = sum(freq for _, freq in sorted_words) if sorted_words else 1
        important_words = [
            {"word": word, "importance": freq / total_freq}
            for word, freq in sorted_words
        ]

        logger.info(f"Extracted important words: {important_words}")
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
            "recommendations": [
                "Verify information from multiple credible sources.",
                "Check for specific citations or expert references.",
                "Be cautious of emotionally charged or sensational language."
            ]
        }

        # Generate summary based on fake probability
        if fake_probability > 0.7:
            explanation["summary"] = "This article contains several characteristics commonly found in fake or misleading news."
        elif fake_probability < 0.3:
            explanation["summary"] = "This article appears to have characteristics of credible news reporting."
        else:
            explanation["summary"] = "This article has mixed characteristics, making it difficult to determine its credibility with high confidence."

        # Generate factors based on features
        if features['sensational_count'] > 0:
            explanation["factors"].append({
                "type": "negative" if features['sensational_count'] > 1 else "neutral",
                "description": f"Contains {features['sensational_count']} sensational terms (e.g., 'mysterious', 'unidentified')."
            })

        if features['credible_count'] > 0:
            explanation["factors"].append({
                "type": "positive" if features['credible_count'] > 1 else "neutral",
                "description": f"Contains {features['credible_count']} credible terms (e.g., 'research', 'scientists')."
            })

        if features['clickbait_count'] > 0:
            explanation["factors"].append({
                "type": "negative",
                "description": f"Contains {features['clickbait_count']} clickbait phrases (e.g., 'never seen before')."
            })

        if features['speculative_count'] > 0:
            explanation["factors"].append({
                "type": "negative",
                "description": f"Contains {features['speculative_count']} speculative phrases (e.g., 'could be', 'may be')."
            })

        if features['exclamation_count'] > 2:
            explanation["factors"].append({
                "type": "negative",
                "description": f"Uses {features['exclamation_count']} exclamation marks, suggesting sensationalism."
            })

        if features['all_caps_count'] > 2:
            explanation["factors"].append({
                "type": "negative",
                "description": f"Contains {features['all_caps_count']} ALL CAPS words, often used for emphasis in fake news."
            })

        if features['suspicious_numbers'] > 0:
            explanation["factors"].append({
                "type": "negative",
                "description": f"Contains {features['suspicious_numbers']} suspicious numerical claims (e.g., 'millions')."
            })

        word_count = features['word_count']
        if word_count < 50:
            explanation["factors"].append({
                "type": "negative",
                "description": f"Short article ({word_count} words), which may lack sufficient detail."
            })
        elif word_count > 200:
            explanation["factors"].append({
                "type": "positive",
                "description": f"Substantial article length ({word_count} words), providing detailed information."
            })

        logger.info(f"Generated explanation: {explanation['summary']}, factors: {len(explanation['factors'])}")
        return explanation

    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        return {
            "summary": "Unable to generate detailed explanation.",
            "factors": [],
            "recommendations": ["Verify information from multiple credible sources."]
        }