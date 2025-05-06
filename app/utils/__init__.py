# Import utility functions for easy access
from app.utils.scraper import scrape_news_from_url, clean_text, extract_metadata
from app.utils.text_processor import preprocess_text, extract_features, extract_important_words, generate_explanation
from app.utils.visualizer import (
    create_word_importance_chart, 
    create_credibility_gauge, 
    create_feature_chart,
    create_wordcloud,
    create_comparison_chart,
    create_feature_comparison_chart,
    create_prediction_distribution_chart
)