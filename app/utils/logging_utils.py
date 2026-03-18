import logging
import json
import uuid
import flask
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "request_id": getattr(flask.g, 'request_id', 'N/A') if flask.has_request_context() else 'N/A'
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def setup_logging(app):
    """Configure structured logging for the application"""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    
    # Root logger configuration
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)
    
    # Flask application logger
    app.logger.handlers = [handler]
    app.logger.setLevel(logging.INFO)

    @app.before_request
    def add_request_id():
        flask.g.request_id = str(uuid.uuid4())

    @app.after_request
    def log_response(response):
        # We could log request/response details here if needed
        return response
