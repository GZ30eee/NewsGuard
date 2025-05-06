import unittest
import json
import sys
import os

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backend import app

class TestBackend(unittest.TestCase):
    """Test cases for the backend API"""
    
    def setUp(self):
        """Set up test client"""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.app.get('/health')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'healthy')
    
    def test_predict_endpoint_valid_input(self):
        """Test prediction endpoint with valid input"""
        payload = {
            "text": "Scientists have discovered a new species of deep-sea fish.",
            "model": "tfidf"
        }
        
        response = self.app.post('/predict', json=payload)
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', data)
        self.assertIn('confidence', data)
        self.assertIn('fake_probability', data)
        self.assertIn('features', data)
        self.assertIn('important_words', data)
        self.assertIn('explanation', data)
    
    def test_predict_endpoint_invalid_input(self):
        """Test prediction endpoint with invalid input"""
        # Missing text
        payload = {
            "model": "tfidf"
        }
        
        response = self.app.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)
        
        # Invalid model
        payload = {
            "text": "Test text",
            "model": "invalid_model"
        }
        
        response = self.app.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)
    
    def test_scrape_endpoint(self):
        """Test scrape endpoint"""
        payload = {
            "url": "https://example.com"
        }
        
        response = self.app.post('/scrape', json=payload)
        
        # Either it succeeds or returns an error about not being able to extract content
        self.assertIn(response.status_code, [200, 400])
    
    def test_batch_predict_endpoint(self):
        """Test batch prediction endpoint"""
        payload = {
            "texts": [
                "Scientists have discovered a new species of deep-sea fish.",
                "BREAKING: Miracle cure for all diseases found in common household item!"
            ],
            "model": "tfidf"
        }
        
        response = self.app.post('/batch-predict', json=payload)
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('results', data)
        self.assertEqual(len(data['results']), 2)

if __name__ == '__main__':
    unittest.main()