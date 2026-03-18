import nltk
import re
import logging
import config
from functools import lru_cache

logger = logging.getLogger(__name__)

# Resource initialization
def init_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words('english'))
    except Exception as e:
        logger.warning(f"Failed to download NLTK resources: {str(e)}")
        return set()

STOP_WORDS = init_resources()

@lru_cache(maxsize=1000)
def preprocess_text(text):
    """
    Clean and tokenize text for model input
    """
    try:
        if not text:
            return ""
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stops and non-alphabetic
        tokens = [token for token in tokens if token not in STOP_WORDS and token.isalpha()]
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        return text

def extract_features(text):
    """
    Extract linguistic and stylistic features from text
    """
    try:
        if not text:
            return _empty_features()
            
        text_lower = text.lower()
        words = text.split()
        
        # Word lists
        sensational_terms = {'breaking', 'shocking', 'miracle', 'secret', '100%', 'cure', 'overnight', 'conspiracy'}
        credible_terms = {'research', 'study', 'scientists', 'published', 'journal', 'evidence', 'analysis'}
        clickbait_phrases = ["you won't believe", "mind blowing", "will shock you", "what happens next"]
        
        # Counts
        sensational_count = sum(1 for term in sensational_terms if term in text_lower)
        credible_count = sum(1 for term in credible_terms if term in text_lower)
        clickbait_count = sum(1 for phrase in clickbait_phrases if phrase in text_lower)
        
        # Stylistic
        exclamation_count = text.count('!')
        question_count = text.count('?')
        all_caps_count = sum(1 for word in words if word.isupper() and len(word) > 2)
        
        # Sentence stats
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        word_count = len(words)
        
        return {
            'sensational_count': sensational_count,
            'credible_count': credible_count,
            'clickbait_count': clickbait_count,
            'exclamation_count': exclamation_count,
            'all_caps_count': all_caps_count,
            'suspicious_numbers': _count_suspicious_numbers(text_lower),
            'word_count': word_count,
            'sentence_count': sentence_count,
            'question_count': question_count
        }
    except Exception as e:
        logger.error(f"Feature extraction error: {str(e)}")
        return _empty_features()

def _count_suspicious_numbers(text):
    patterns = [r'\d+\s*%\s*effective', r'lose\s*\d+\s*pounds', r'\d+\s*simple\s*tricks']
    return sum(len(re.findall(p, text)) for p in patterns)

def _empty_features():
    return {
        'sensational_count': 0, 'credible_count': 0, 'clickbait_count': 0,
        'exclamation_count': 0, 'all_caps_count': 0, 'suspicious_numbers': 0,
        'word_count': 0, 'sentence_count': 0, 'question_count': 0
    }

def generate_explanation(text, features, fake_probability):
    """
    Generate human-readable explanation for predictions
    """
    explanation = {
        "summary": "",
        "factors": [],
        "recommendations": [
            "Verify information from multiple credible sources.",
            "Check for specific citations or expert references.",
            "Be cautious of emotionally charged language."
        ]
    }

    # Summary
    if fake_probability > config.FAKE_THRESHOLD:
        explanation["summary"] = "This article contains several characteristics common in misleading news."
    elif fake_probability < config.REAL_THRESHOLD:
        explanation["summary"] = "This article appears to follow credible reporting standards."
    else:
        explanation["summary"] = "This article has mixed signals, indicating uncertainty."

    # Factors
    if features['sensational_count'] > 0:
        explanation["factors"].append({
            "type": "negative",
            "description": f"Contains {features['sensational_count']} sensational terms designed to trigger emotional responses."
        })
    
    if features['credible_count'] > 0:
        explanation["factors"].append({
            "type": "positive",
            "description": f"Uses {features['credible_count']} terms associated with evidence-based reporting."
        })

    if features['all_caps_count'] > 2:
        explanation["factors"].append({
            "type": "negative",
            "description": f"Uses ALL CAPS for {features['all_caps_count']} words, which is typical of sensationalist content."
        })

    return explanation

def extract_important_words(text, n=10):
    """
    Identify key words for visualization
    """
    try:
        processed = preprocess_text(text)
        if not processed:
            return []
        
        tokens = processed.split()
        from collections import Counter
        counts = Counter(tokens)
        top = counts.most_common(n)
        
        total = sum(c for _, c in top) or 1
        return [{"word": w, "importance": float(c / total)} for w, c in top]
    except Exception as e:
        logger.error(f"Important words extraction error: {str(e)}")
        return []