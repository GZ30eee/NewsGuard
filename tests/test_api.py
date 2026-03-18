import unittest
import json
from app.backend import app
import config

class NewsGuardApiTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertIn('healthy', response.text)

    def test_predict_tfidf(self):
        data = {
            "text": "Scientists discover a new species of fish in the deep ocean.",
            "model": "tfidf"
        }
        response = self.app.post('/api/predict', 
                                 data=json.dumps(data),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        res_json = response.json
        self.assertIn('prediction', res_json)
        self.assertIn('confidence', res_json)

    def test_predict_validation(self):
        # Empty text
        response = self.app.post('/api/predict', 
                                 data=json.dumps({"text": ""}),
                                 content_type='application/json')
        # Service might handle empty text or return error. Let's check status
        self.assertEqual(response.status_code, 200) # Sanitizer returns empty string

    def test_scrape_validation(self):
        # SSRF attempt
        data = {"url": "http://127.0.0.1:8080/admin"}
        response = self.app.post('/api/scrape', 
                                 data=json.dumps(data),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 403)
        self.assertIn('Insecure', response.json['error'])

if __name__ == '__main__':
    unittest.main()
