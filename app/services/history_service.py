import os
import uuid
import json
import logging
from datetime import datetime
import config

logger = logging.getLogger(__name__)

class HistoryService:
    @staticmethod
    def save_analysis(result):
        """Save analysis result to a JSON file"""
        try:
            analysis_id = result.get('id', str(uuid.uuid4()))
            result['id'] = analysis_id
            result['timestamp'] = datetime.now().isoformat()
            
            filename = f"{analysis_id}.json"
            filepath = os.path.join(config.HISTORY_DIR, filename)
            
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=4)
            
            logger.info(f"Saved analysis {analysis_id}")
            return analysis_id
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            return None

    @staticmethod
    def get_history(limit=50):
        """Retrieve analysis history"""
        history = []
        try:
            files = os.listdir(config.HISTORY_DIR)
            # Filter for .json files and sort by modification time (newest first)
            json_files = [f for f in files if f.endswith('.json')]
            json_files.sort(key=lambda x: os.path.getmtime(os.path.join(config.HISTORY_DIR, x)), reverse=True)
            
            for filename in json_files[:limit]:
                try:
                    with open(os.path.join(config.HISTORY_DIR, filename), 'r') as f:
                        data = json.load(f)
                        # Remove full text to save bandwidth in list view
                        if 'text' in data:
                            data['text_preview'] = data['text'][:200] + "..."
                            del data['text']
                        history.append(data)
                except Exception as e:
                    logger.error(f"Error reading history file {filename}: {str(e)}")
                    
            return history
        except Exception as e:
            logger.error(f"Error retrieving history: {str(e)}")
            return []

    @staticmethod
    def get_analysis_by_id(analysis_id):
        """Get a specific analysis by ID"""
        try:
            filename = f"{analysis_id}.json"
            filepath = os.path.join(config.HISTORY_DIR, filename)
            
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error retrieving analysis {analysis_id}: {str(e)}")
            return None

    @staticmethod
    def delete_analysis(analysis_id):
        """Delete a specific analysis by ID"""
        try:
            filename = f"{analysis_id}.json"
            filepath = os.path.join(config.HISTORY_DIR, filename)
            
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Deleted analysis {analysis_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting analysis {analysis_id}: {str(e)}")
            return False
