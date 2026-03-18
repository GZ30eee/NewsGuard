from flask import Blueprint, request, jsonify, current_app
from app.services.history_service import HistoryService
from app.utils.validation import is_safe_url, sanitize_text
from app.utils.scraper import scrape_all
import logging

news_bp = Blueprint('news', __name__)
logger = logging.getLogger(__name__)

@news_bp.route('/predict', methods=['POST'])
def predict():
    """Predict if a news article is fake or real"""
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' in request body"}), 400
        
        text = sanitize_text(data['text'])
        model_type = data.get('model', 'tfidf')
        
        # Use the prediction service from app context
        prediction_service = current_app.prediction_service
        
        if model_type == 'ensemble':
            result = prediction_service.ensemble_predict(text)
        else:
            result = prediction_service.predict(text, model_type)
            
        # Save to history automatically
        HistoryService.save_analysis(result)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction route error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error during prediction"}), 500

@news_bp.route('/scrape', methods=['POST'])
def scrape():
    """Scrape article text and metadata from a URL"""
    try:
        data = request.json
        if not data or 'url' not in data:
            return jsonify({"error": "Missing 'url' in request body"}), 400
        
        url = data['url']
        if not is_safe_url(url):
            return jsonify({"error": "Insecure or invalid URL (blocked for SSRF protection)"}), 403
            
        # Use scrape_all to do both in one request
        result = scrape_all(url)
        
        if not result or not result.get('text'):
            return jsonify({"error": "Failed to extract meaningful content from URL. Site might be blocking automated access."}), 404
            
        return jsonify(result)
    except Exception as e:
        logger.error(f"Scrape route error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Server encountered an error while scraping: {str(e)}"}), 500

@news_bp.route('/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    try:
        history = HistoryService.get_history()
        return jsonify(history)
    except Exception as e:
        logger.error(f"History route error: {str(e)}")
        return jsonify({"error": "Failed to retrieve history"}), 500

@news_bp.route('/history/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """Get specific analysis result"""
    result = HistoryService.get_analysis_by_id(analysis_id)
    if not result:
        return jsonify({"error": "Analysis not found"}), 404
    return jsonify(result)

@news_bp.route('/history/<analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    """Delete specific analysis result"""
    if HistoryService.delete_analysis(analysis_id):
        return jsonify({"message": "Deleted successfully"})
    return jsonify({"error": "Analysis not found"}), 404

@news_bp.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Predict multiple articles in one request"""
    try:
        data = request.json
        if not data or 'articles' not in data or not isinstance(data['articles'], list):
            return jsonify({"error": "Invalid request: 'articles' must be a list"}), 400
        
        model_type = data.get('model', 'tfidf')
        prediction_service = current_app.prediction_service
        
        results = []
        for article in data['articles']:
            text = sanitize_text(article.get('text', ''))
            if not text:
                continue
            
            try:
                res = prediction_service.predict(text, model_type)
                res['title'] = article.get('title', 'Untitled')
                results.append(res)
            except Exception as e:
                logger.warning(f"Batch item failed: {str(e)}")
                
        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({"error": "Batch processing failed"}), 500
