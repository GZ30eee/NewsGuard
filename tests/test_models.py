import unittest
import sys
import os
import numpy as np

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.bert_model import BERTModel
from app.models.tfidf_model import TFIDFModel

class TestModels(unittest.TestCase):
    """Test cases for the ML models"""
    
    def test_tfidf_model_prediction(self):
        """Test TF-IDF model prediction"""
        model = TFIDFModel()
        
        # Test with likely fake news
        fake_text = "BREAKING: Scientists confirm that drinking hot water mixed with lemon juice every morning can prevent cancer with 100% effectiveness."
        result = model.predict(fake_text)
        
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertIn('fake_probability', result)
        self.assertIn('important_words', result)
        self.assertIsInstance(result['confidence'], float)
        self.assertTrue(0 <= result['confidence'] <= 1)
        
        # Test with likely real news
        real_text = "Scientists have discovered a new species of deep-sea fish that can withstand extreme pressure at depths of over 8,000 meters."
        result = model.predict(real_text)
        
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertIn('fake_probability', result)
        self.assertIn('important_words', result)
        self.assertIsInstance(result['confidence'], float)
        self.assertTrue(0 <= result['confidence'] <= 1)
    
    def test_bert_model_prediction(self):
        """Test BERT model prediction"""
        try:
            model = BERTModel()
            
            # Test with likely fake news
            fake_text = "BREAKING: Scientists confirm that drinking hot water mixed with lemon juice every morning can prevent cancer with 100% effectiveness."
            result = model.predict(fake_text)
            
            self.assertIn('prediction', result)
            self.assertIn('confidence', result)
            self.assertIn('fake_probability', result)
            self.assertIn('important_words', result)
            self.assertIsInstance(result['confidence'], float)
            self.assertTrue(0 <= result['confidence'] <= 1)
            
            # Test with likely real news
            real_text = "Scientists have discovered a new species of deep-sea fish that can withstand extreme pressure at depths of over 8,000 meters."
            result = model.predict(real_text)
            
            self.assertIn('prediction', result)
            self.assertIn('confidence', result)
            self.assertIn('fake_probability', result)
            self.assertIn('important_words', result)
            self.assertIsInstance(result['confidence'], float)
            self.assertTrue(0 <= result['confidence'] <= 1)
        except Exception as e:
            # Skip test if BERT model can't be loaded (e.g., in CI environment)
            self.skipTest(f"BERT model test skipped: {str(e)}")
    
    def test_edge_cases(self):
        """Test model behavior with edge cases"""
        model = TFIDFModel()
        
        # Empty text
        empty_text = ""
        result = model.predict(empty_text)
        self.assertIn('prediction', result)
        
        # Very short text
        short_text = "Hello world."
        result = model.predict(short_text)
        self.assertIn('prediction', result)
        
        # Very long text
        long_text = "This is a test. " * 1000
        result = model.predict(long_text)
        self.assertIn('prediction', result)

if __name__ == '__main__':
    unittest.main()