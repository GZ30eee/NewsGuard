import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np
import pandas as pd
import io
import base64

def create_word_importance_chart(important_words):
    """
    Create a bar chart of word importance
    
    Args:
        important_words (list): List of dictionaries with 'word' and 'importance' keys
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if not important_words:
        return None
    
    # Extract words and importance scores
    words = [item["word"] for item in important_words]
    importance = [item["importance"] for item in important_words]
    
    # Create bar chart
    fig = px.bar(
        x=importance,
        y=words,
        orientation='h',
        labels={"x": "Importance", "y": "Word"},
        title="Words by Importance"
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        xaxis_title="Importance Score",
        yaxis_title="Word",
        yaxis=dict(autorange="reversed")
    )
    
    return fig

def create_credibility_gauge(credibility_score):
    """
    Create a gauge chart for credibility score
    
    Args:
        credibility_score (float): Credibility score (0-100)
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = credibility_score,
        title = {'text': "Credibility Score"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "red"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': credibility_score
            }
        }
    ))
    
    # Update layout
    fig.update_layout(height=300)
    
    return fig

def create_feature_chart(features):
    """
    Create a chart visualizing feature values
    
    Args:
        features (dict): Dictionary of feature values
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Select relevant features
    feature_names = [
        "Sensational Terms",
        "Credible Terms",
        "Clickbait Phrases",
        "Exclamation Marks",
        "Question Marks",
        "ALL CAPS Words",
        "Suspicious Numbers"
    ]
    
    feature_values = [
        features["sensational_count"],
        features["credible_count"],
        features["clickbait_count"],
        features["exclamation_count"],
        features["question_count"],
        features["all_caps_count"],
        features["suspicious_numbers"]
    ]
    
    # Normalize values for better visualization
    max_val = max(feature_values) if max(feature_values) > 0 else 1
    normalized_values = [val / max_val for val in feature_values]
    
    # Create color map (red for negative, green for positive)
    colors = ['red', 'green', 'red', 'red', 'yellow', 'red', 'red']
    
    # Create bar chart
    fig = px.bar(
        x=normalized_values,
        y=feature_names,
        orientation='h',
        labels={"x": "Normalized Value", "y": "Feature"},
        title="Feature Analysis"
    )
    
    # Update colors
    for i, color in enumerate(colors):
        fig.data[0].marker.color = colors
    
    # Update layout
    fig.update_layout(height=400)
    
    return fig

def create_wordcloud(text, dark_mode=False):
    """
    Create a word cloud from text
    
    Args:
        text (str): Input text
        dark_mode (bool): Whether to use dark mode colors
        
    Returns:
        str: Base64 encoded image
    """
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='black' if dark_mode else 'white',
        colormap='viridis',
        max_words=100,
        contour_width=1
    ).generate(text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Encode as base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def create_comparison_chart(comparison_data):
    """
    Create a chart comparing multiple articles
    
    Args:
        comparison_data (list): List of article data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create dataframe
    df = pd.DataFrame([
        {
            "ID": item["id"][:8],
            "Prediction": item["prediction"],
            "Confidence": item["confidence"],
            "Credibility Score": (1 - item["fake_probability"]) * 100,
            "Sensational Terms": item["features"]["sensational_count"],
            "Credible Terms": item["features"]["credible_count"],
            "Clickbait Phrases": item["features"]["clickbait_count"]
        }
        for item in comparison_data
    ])
    
    # Create bar chart
    fig = px.bar(
        df,
        x="ID",
        y="Credibility Score",
        color="Prediction",
        color_discrete_map={
            "FAKE": "red",
            "REAL": "green",
            "UNCERTAIN": "yellow"
        },
        labels={"Credibility Score": "Credibility Score (0-100)", "ID": "Article ID"},
        title="Credibility Score Comparison"
    )
    
    return fig

def create_feature_comparison_chart(comparison_data, features):
    """
    Create a chart comparing specific features across multiple articles
    
    Args:
        comparison_data (list): List of article data
        features (list): List of feature names to compare
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Map feature names to keys
    feature_map = {
        "Sensational Terms": "sensational_count",
        "Credible Terms": "credible_count",
        "Clickbait Phrases": "clickbait_count",
        "Exclamation Marks": "exclamation_count",
        "Question Marks": "question_count",
        "ALL CAPS Words": "all_caps_count",
        "Suspicious Numbers": "suspicious_numbers"
    }
    
    # Add traces for each feature
    for feature in features:
        feature_key = feature_map.get(feature)
        if not feature_key:
            continue
            
        fig.add_trace(go.Bar(
            x=[item["id"][:8] for item in comparison_data],
            y=[item["features"][feature_key] for item in comparison_data],
            name=feature
        ))
    
    # Update layout
    fig.update_layout(
        title="Feature Comparison",
        xaxis_title="Article ID",
        yaxis_title="Count",
        barmode='group'
    )
    
    return fig

def create_prediction_distribution_chart(results_df):
    """
    Create a pie chart showing the distribution of predictions
    
    Args:
        results_df (pandas.DataFrame): DataFrame with prediction results
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Count predictions
    prediction_counts = results_df["Prediction"].value_counts()
    
    # Create pie chart
    fig = px.pie(
        values=prediction_counts.values,
        names=prediction_counts.index,
        title="Prediction Distribution",
        color=prediction_counts.index,
        color_discrete_map={
            "FAKE": "red",
            "REAL": "green",
            "UNCERTAIN": "yellow"
        }
    )
    
    return fig