import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import base64
from io import StringIO
import time
import os
import uuid
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoint (adjust if needed)
API_URL = "http://localhost:5000"

# Set page configuration
st.set_page_config(
    page_title="NewsGuard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []
if 'comparison_items' not in st.session_state:
    st.session_state.comparison_items = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'analysis_id' not in st.session_state:
    st.session_state.analysis_id = None

# Custom CSS for styling
def local_css():
    st.markdown("""
    <style>
        .main {
            background-color: var(--background-color);
            color: var(--text-color);
            transition: all 0.5s ease;
        }
        .stTextInput, .stTextArea {
            background-color: var(--input-bg);
            color: var(--text-color);
            border-radius: 5px;
        }
        .stButton>button {
            background-color: var(--button-bg);
            color: var(--button-text);
        }
        .fake-news-result {
            background-color: var(--fake-bg);
            color: var(--fake-text);
            padding: 20px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .real-news-result {
            background-color: var(--real-bg);
            color: var(--real-text);
            padding: 20px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .uncertain-news-result {
            background-color: var(--uncertain-bg);
            color: var(--uncertain-text);
            padding: 20px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .feature-card {
            background-color: var(--card-bg);
            padding: 20px;
            border-radius: 5px;
            margin: 10px 0;
            border: 1px solid var(--card-border);
        }
        .history-item {
            background-color: var(--card-bg);
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            border: 1px solid var(--card-border);
        }
    </style>
    """, unsafe_allow_html=True)

# Apply appropriate theme based on dark mode setting
def apply_theme():
    if st.session_state.dark_mode:
        # Dark theme variables
        st.markdown("""
        <style>
        :root {
            --background-color: #121212;
            --text-color: #f0f0f0;
            --input-bg: #2d2d2d;
            --button-bg: #4CAF50;
            --button-text: white;
            --fake-bg: #8B0000;
            --fake-text: white;
            --real-bg: #006400;
            --real-text: white;
            --uncertain-bg: #856404;
            --uncertain-text: white;
            --card-bg: #2D2D2D;
            --card-border: #444444;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Light theme variables
        st.markdown("""
        <style>
        :root {
            --background-color: #ffffff;
            --text-color: #333333;
            --input-bg: #f9f9f9;
            --button-bg: #4CAF50;
            --button-text: white;
            --fake-bg: #FFCCCC;
            --fake-text: #8B0000;
            --real-bg: #CCFFCC;
            --real-text: #006400;
            --uncertain-bg: #FFF3CD;
            --uncertain-text: #856404;
            --card-bg: #F8F9FA;
            --card-border: #DEE2E6;
        }
        </style>
        """, unsafe_allow_html=True)

# Toggle dark mode
def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Function to make prediction request to backend
def predict_news(text, model_choice):
    """
    Send prediction request to backend API
    
    Args:
        text (str): News article text
        model_choice (str): Model to use ('bert' or 'tfidf')
        
    Returns:
        dict: Prediction result or None if error
    """
    try:
        # Prepare the request payload
        payload = {
            "text": text,
            "model": model_choice.lower().replace("-", "")  # Remove hyphen
        }
        
        # Send request to backend
        with st.spinner(f"Analyzing with {model_choice.upper()} model..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Add to session history
                if result not in st.session_state.history:
                    st.session_state.history.append(result)
                    # Limit history to 20 items
                    if len(st.session_state.history) > 20:
                        st.session_state.history.pop(0)
                
                return result
            else:
                error_msg = "Unknown error"
                try:
                    error_msg = response.json().get('error', 'Unknown error')
                except:
                    pass
                st.error(f"Error: {error_msg}")
                return None
                
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {str(e)}")
        st.info("Make sure the backend server is running at " + API_URL)
        return None
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Function to scrape news from URL
def scrape_url(url):
    """
    Send scrape request to backend API
    
    Args:
        url (str): URL to scrape
        
    Returns:
        str: Scraped text or None if error
    """
    try:
        # Prepare the request payload
        payload = {
            "url": url
        }
        
        # Send request to backend
        with st.spinner(f"Fetching content from {url}..."):
            response = requests.post(f"{API_URL}/scrape", json=payload, timeout=30)
            
            # Check if request was successful
            if response.status_code == 200:
                return response.json().get("text")
            else:
                error_msg = "Unknown error"
                try:
                    error_msg = response.json().get('error', 'Unknown error')
                except:
                    pass
                st.error(f"Error: {error_msg}")
                return None
                
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {str(e)}")
        st.info("Make sure the backend server is running at " + API_URL)
        return None
    except Exception as e:
        st.error(f"Error scraping URL: {str(e)}")
        return None

# Function to handle batch prediction
def process_batch_file(df, model_choice):
    """
    Process batch prediction
    
    Args:
        df (pandas.DataFrame): DataFrame with 'text' column
        model_choice (str): Model to use ('bert' or 'tfidf')
        
    Returns:
        list: List of prediction results or None if error
    """
    try:
        # Prepare the request payload
        payload = {
            "texts": df["text"].tolist(),
            "model": model_choice.lower()
        }
        
        # Send request to backend
        with st.spinner(f"Processing batch with {model_choice.upper()} model..."):
            response = requests.post(f"{API_URL}/batch-predict", json=payload, timeout=120)
            
            # Check if request was successful
            if response.status_code == 200:
                results = response.json().get("results", [])
                
                # Add to session history
                for result in results:
                    if result not in st.session_state.history:
                        st.session_state.history.append(result)
                        # Limit history to 20 items
                        if len(st.session_state.history) > 20:
                            st.session_state.history.pop(0)
                
                return results
            else:
                error_msg = "Unknown error"
                try:
                    error_msg = response.json().get('error', 'Unknown error')
                except:
                    pass
                st.error(f"Error: {error_msg}")
                return None
                
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {str(e)}")
        st.info("Make sure the backend server is running at " + API_URL)
        return None
    except Exception as e:
        st.error(f"Error processing batch: {str(e)}")
        return None

# Function to get analysis history
def get_history():
    """
    Get analysis history from backend API
    
    Returns:
        list: List of history items or empty list if error
    """
    try:
        # Send request to backend
        response = requests.get(f"{API_URL}/history", timeout=10)
        
        # Check if request was successful
        if response.status_code == 200:
            history = response.json().get("history", [])
            
            # Update session history
            st.session_state.history = history
            
            return history
        else:
            error_msg = "Unknown error"
            try:
                error_msg = response.json().get('error', 'Unknown error')
            except:
                pass
            st.error(f"Error: {error_msg}")
            return []
                
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {str(e)}")
        st.info("Make sure the backend server is running at " + API_URL)
        return []
    except Exception as e:
        st.error(f"Error getting history: {str(e)}")
        return []

# Function to add item to comparison
def add_to_comparison(result):
    """
    Add an item to the comparison list
    
    Args:
        result (dict): Analysis result to add
    
    Returns:
        bool: True if added successfully, False otherwise
    """
    try:
        # Check if already in comparison
        if not any(item.get("id") == result.get("id") for item in st.session_state.comparison_items):
            # Make a deep copy to avoid reference issues
            st.session_state.comparison_items.append(result.copy())
            return True
        return False
    except Exception as e:
        logger.error(f"Error adding to comparison: {str(e)}")
        return False

# Function to display analysis results
def display_analysis_results(result):
    """
    Display analysis results in the UI
    
    Args:
        result (dict): Analysis result
    """
    if not result:
        st.error("No analysis result to display")
        return
        
    # Set the analysis ID in session state
    st.session_state.analysis_id = result.get("id")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display prediction with appropriate styling
        if result.get("prediction") == "FAKE":
            st.markdown(f"""
            <div class="fake-news-result">
                <h2>⚠️ Likely Fake News</h2>
                <p>This article has characteristics commonly found in misleading or false information.</p>
                <h3>Confidence: {result.get("confidence", 0)*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        elif result.get("prediction") == "REAL":
            st.markdown(f"""
            <div class="real-news-result">
                <h2>✅ Likely Real News</h2>
                <p>This article appears to have characteristics of credible reporting.</p>
                <h3>Confidence: {result.get("confidence", 0)*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="uncertain-news-result">
                <h2>⚖️ Uncertain</h2>
                <p>This article has mixed characteristics, making it difficult to determine its credibility with high confidence.</p>
                <h3>Confidence: {result.get("confidence", 0)*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Display explanation
        st.subheader("Analysis Explanation")
        explanation = result.get("explanation", {})
        st.write(explanation.get("summary", "No explanation available"))
        
        # Display factors
        st.write("**Key Factors:**")
        for factor in explanation.get("factors", []):
            if factor.get("type") == "positive":
                st.markdown(f"✅ {factor.get('description', '')}")
            elif factor.get("type") == "negative":
                st.markdown(f"⚠️ {factor.get('description', '')}")
            else:
                st.markdown(f"ℹ️ {factor.get('description', '')}")
        
        # Display recommendations
        st.subheader("Recommendations")
        for recommendation in explanation.get("recommendations", []):
            st.markdown(f"- {recommendation}")
    
    with col2:
        # Display credibility score gauge
        fake_probability = result.get("fake_probability", 0.5)
        credibility_score = (1 - fake_probability) * 100
        
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
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display important words
        st.subheader("Key Words Analysis")
        
        # Create word importance chart
        important_words = result.get("important_words", [])
        if important_words:
            words = [item.get("word", "") for item in important_words]
            importance = [item.get("importance", 0) for item in important_words]
            
            fig = px.bar(
                x=importance,
                y=words,
                orientation='h',
                labels={"x": "Importance", "y": "Word"},
                title="Words by Importance"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No important words identified")
        
        # Add to comparison button
        # if st.button("Add to Comparison"):
        #     if add_to_comparison(result):
        #         st.success("Added to comparison!")
        #     else:
        #         st.info("This article is already in the comparison list.")
    
    # Display word cloud
    st.subheader("Word Cloud Visualization")
    
    # Generate word cloud
    try:
        text = result.get("text", "")
        if text:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='black' if st.session_state.dark_mode else 'white',
                max_words=100,
                colormap='viridis',
                contour_width=1
            ).generate(text)
            
            # Display word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No text available for word cloud")
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")
    
    # Display detailed feature analysis
    with st.expander("Detailed Feature Analysis"):
        # Create feature comparison chart
        features = result.get("features", {})
        if features:
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
                features.get("sensational_count", 0),
                features.get("credible_count", 0),
                features.get("clickbait_count", 0),
                features.get("exclamation_count", 0),
                features.get("question_count", 0),
                features.get("all_caps_count", 0),
                features.get("suspicious_numbers", 0)
            ]
            
            # Normalize values for better visualization
            max_val = max(feature_values) if max(feature_values) > 0 else 1
            normalized_values = [val / max_val for val in feature_values]
            
            # Create color map (red for negative, green for positive)
            colors = ['red', 'green', 'red', 'red', 'yellow', 'red', 'red']
            
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
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display raw feature values
            st.write("Raw Feature Values:")
            feature_df = pd.DataFrame({
                "Feature": feature_names,
                "Value": feature_values
            })
            st.dataframe(feature_df)
        else:
            st.info("No feature data available")
    
    # Export options
    with st.expander("Export Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as JSON
            json_data = json.dumps(result, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"fake_news_analysis_{result.get('id', 'export')}.json",
                mime="application/json"
            )
        
        with col2:
            # Export as CSV
            text = result.get("text", "").replace('"', '""')  # Escape quotes for CSV
            prediction = result.get("prediction", "")
            confidence = result.get("confidence", 0)
            fake_probability = result.get("fake_probability", 0)
            
            features = result.get("features", {})
            sensational_count = features.get("sensational_count", 0)
            credible_count = features.get("credible_count", 0)
            clickbait_count = features.get("clickbait_count", 0)
            
            csv_data = f"""text,prediction,confidence,fake_probability,sensational_count,credible_count,clickbait_count
"{text}",{prediction},{confidence},{fake_probability},{sensational_count},{credible_count},{clickbait_count}
"""
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"fake_news_analysis_{result.get('id', 'export')}.csv",
                mime="text/csv"
            )

# Function to display comparison view
def display_comparison_view():
    """Display comparison view for multiple articles"""
    if not st.session_state.comparison_items:
        st.info("No items added to comparison yet. Analyze articles and click 'Add to Comparison' to compare them.")
        return
    
    st.subheader(f"Comparing {len(st.session_state.comparison_items)} Articles")
    
    # Create dataframe for comparison
    comparison_data = []
    for item in st.session_state.comparison_items:
        features = item.get("features", {})
        comparison_data.append({
            "ID": item.get("id", "")[:8],
            "Prediction": item.get("prediction", ""),
            "Confidence": item.get("confidence", 0),
            "Credibility Score": (1 - item.get("fake_probability", 0)) * 100,
            "Sensational Terms": features.get("sensational_count", 0),
            "Credible Terms": features.get("credible_count", 0),
            "Clickbait Phrases": features.get("clickbait_count", 0),
            "Text Preview": item.get("text", "")[:100] + "..." if len(item.get("text", "")) > 100 else item.get("text", "")
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.dataframe(comparison_df)
    
    # Create comparison chart
    st.subheader("Credibility Score Comparison")
    
    fig = px.bar(
        comparison_df,
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
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature comparison
    st.subheader("Feature Comparison")
    
    # Select features to compare
    feature_options = ["Sensational Terms", "Credible Terms", "Clickbait Phrases"]
    selected_features = st.multiselect(
        "Select features to compare",
        feature_options,
        default=feature_options
    )
    
    if selected_features:
        # Create feature comparison chart
        fig = go.Figure()
        
        for feature in selected_features:
            fig.add_trace(go.Bar(
                x=comparison_df["ID"],
                y=comparison_df[feature],
                name=feature
            ))
        
        fig.update_layout(
            title="Feature Comparison",
            xaxis_title="Article ID",
            yaxis_title="Count",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # View details button
    selected_id = st.selectbox(
        "Select an article to view details",
        options=comparison_df["ID"].tolist(),
        format_func=lambda x: f"{x} - {comparison_df[comparison_df['ID'] == x]['Text Preview'].iloc[0]}"
    )
    
    if selected_id and st.button("View Selected Article Details"):
        # Find the selected item
        selected_item = next((item for item in st.session_state.comparison_items if item.get("id", "").startswith(selected_id)), None)
        
        if selected_item:
            # Display the selected item
            st.subheader("Article Details")
            display_analysis_results(selected_item)
    
    # Clear comparison button
    if st.button("Clear Comparison"):
        st.session_state.comparison_items = []
        st.success("Comparison cleared!")
        st.rerun()

# Function to display history view
def display_history_view():
    """Display history of previous analyses"""
    # Get history from backend or use session state
    history = get_history() if not st.session_state.history else st.session_state.history
    
    if not history:
        st.info("No analysis history yet. Analyze some articles to see them here.")
        return
    
    st.subheader("Analysis History")
    
    # Create dataframe for history
    history_data = []
    for item in history:
        history_data.append({
            "ID": item.get("id", "")[:8],
            "Timestamp": item.get("timestamp", ""),
            "Prediction": item.get("prediction", ""),
            "Confidence": item.get("confidence", 0),
            "Text Preview": item.get("text", "")[:100] + "..." if len(item.get("text", "")) > 100 else item.get("text", "")
        })
    
    history_df = pd.DataFrame(history_data)
    
    # Sort by timestamp (newest first)
    if "Timestamp" in history_df.columns and not history_df["Timestamp"].empty:
        try:
            history_df["Timestamp"] = pd.to_datetime(history_df["Timestamp"])
            history_df = history_df.sort_values("Timestamp", ascending=False)
        except:
            pass
    
    # Display history table
    st.dataframe(history_df)
    
    # Select item to view
    if not history_df.empty:
        selected_id = st.selectbox(
            "Select an item to view details",
            options=history_df["ID"].tolist(),
            format_func=lambda x: f"{x} - {history_df[history_df['ID'] == x]['Text Preview'].iloc[0]}"
        )
        
        if selected_id and st.button("View Selected Analysis"):
            # Find the selected item
            selected_item = next((item for item in history if item.get("id", "").startswith(selected_id)), None)
            
            if selected_item:
                # Display the selected item
                st.subheader("Analysis Details")
                display_analysis_results(selected_item)
                
                # Add to comparison button
                if st.button("Add This Analysis to Comparison"):
                    if add_to_comparison(selected_item):
                        st.success("Added to comparison!")
                    else:
                        st.info("This article is already in the comparison list.")

# Main function
def main():
    # Apply theme and CSS
    apply_theme()
    local_css()
    
    # Sidebar
    with st.sidebar:
        st.title("🔍 NewsGuard")
        
        # Dark mode toggle
        # st.checkbox("Dark Mode", value=st.session_state.dark_mode, on_change=toggle_dark_mode)
        
        # Model selection
        st.subheader("Model Selection")
        model_choice = st.radio(
            "Choose prediction model:",
            ["BERT", "TF-IDF"],
            format_func=lambda x: f"{x} ({'High Accuracy' if x == 'BERT' else 'Fast'})"
        )
        
        # Model information
        with st.expander("Model Information"):
            if model_choice == "BERT":
                st.write("""
                **BERT Model**
                - Based on DistilBERT architecture
                - Higher accuracy but slower prediction
                - Better at understanding context
                - Fine-tuned on fake news dataset
                """)
            else:
                st.write("""
                **TF-IDF + Logistic Regression**
                - Classic machine learning approach
                - Faster prediction but less accurate
                - Uses word frequency features
                - Good for quick analysis
                """)
        
        # Navigation
        st.subheader("Navigation")
        page = st.radio(
            "Go to",
            ["Text Analysis", "URL Analysis", "Batch Analysis", "History"]
        )
        
        # About section
        with st.expander("About"):
            st.write("""
            **NewsGuard**
            
            This application uses machine learning to detect potentially fake news articles.
            
            - Input text directly or via URL
            - Choose between two ML models
            - Get prediction with confidence score
            - See which words influenced the prediction
            - Process multiple articles via CSV upload
            
            *Note: No ML model is perfect. Always verify information from multiple reliable sources.*
            """)
        
        # Check backend connection
            # response = requests.get(f"{API_URL}/health", timeout=2)
            # if response.status_code == 200:
            #     st.success("✅ Backend connected")
            # else:
            #     st.error("❌ Backend error")
            # st.error("❌ Backend not connected")
            # st.info(f"Make sure the backend server is running at {API_URL}")
    
    # Main content
    if page == "Text Analysis":
        st.title("Text Analysis")
        st.write("Analyze news articles to determine if they might be fake or real using machine learning.")
        
        # Sample articles
        with st.expander("Sample Articles"):
            sample_options = {
                "Select a sample": "",
                "Likely Fake News": "BREAKING: Scientists confirm that drinking hot water mixed with lemon juice every morning can prevent cancer with 100% effectiveness. This revolutionary discovery has been suppressed by pharmaceutical companies for decades because it would eliminate the need for expensive cancer treatments. Share this information before it gets taken down!",
                "Likely Real News": "Researchers at Stanford University have developed a new algorithm that improves early detection of Alzheimer's disease. The study, published in the Journal of Medical AI, shows a 40% improvement in early diagnosis compared to current methods. The team analyzed brain scans from over 5,000 patients using machine learning techniques.",
                "Ambiguous News": "A new study suggests that coffee might have additional health benefits. Some experts believe it could reduce the risk of certain conditions, while others remain skeptical. The research is still in early stages and more investigation is needed to confirm these preliminary findings."
            }
            
            sample_selection = st.selectbox(
                "Choose a sample article",
                options=list(sample_options.keys())
            )
            
            if sample_selection != "Select a sample":
                st.info(f"Selected: {sample_selection}")
                st.write(sample_options[sample_selection])
                
                # Use sample button
                if st.button("Use this sample"):
                    st.session_state.sample_text = sample_options[sample_selection]
                    st.rerun()
        
        # Text input
        if 'sample_text' in st.session_state:
            news_text = st.text_area("News Article Text", value=st.session_state.sample_text, height=200)
            # Clear the sample after using it
            del st.session_state.sample_text
        else:
            news_text = st.text_area("News Article Text", height=200, placeholder="Paste or type a news article here...")
        
        # Analyze button
        if st.button("Analyze Article"):
            if news_text.strip():
                # Send to backend for analysis
                result = predict_news(news_text, model_choice)
                
                if result:
                    # Display results
                    display_analysis_results(result)
            else:
                st.warning("Please enter some text to analyze.")
    
    elif page == "URL Analysis":
        st.title("URL Analysis")
        st.write("Enter a URL to fetch and analyze a news article.")
        
        # URL input
        news_url = st.text_input("News Article URL", placeholder="https://example.com/news-article")
        
        # Analyze button
        if st.button("Fetch & Analyze"):
            if news_url.strip():
                # Scrape the URL
                article_text = scrape_url(news_url)
                
                if article_text:
                    st.success("Article fetched successfully!")
                    
                    # Display the extracted text
                    with st.expander("Extracted Content", expanded=True):
                        st.write(article_text)
                    
                    # Send to backend for analysis
                    result = predict_news(article_text, model_choice)
                    
                    if result:
                        # Display results
                        display_analysis_results(result)
            else:
                st.warning("Please enter a URL to analyze.")
    
    elif page == "Batch Analysis":
        st.title("Batch Analysis")
        st.write("Upload a CSV file with a 'text' column to analyze multiple articles at once.")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV File", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                # Check if 'text' column exists
                if 'text' not in df.columns:
                    st.error("CSV file must contain a 'text' column with news articles.")
                else:
                    # Display preview
                    st.subheader("Preview of Uploaded Data")
                    st.dataframe(df.head())
                    
                    # Process button
                    if st.button("Process Batch"):
                        # Send to backend for batch analysis
                        results = process_batch_file(df, model_choice)
                        
                        if results:
                            # Create results dataframe
                            results_df = pd.DataFrame([
                                {
                                    "Text": item.get("text", "")[:100] + "..." if len(item.get("text", "")) > 100 else item.get("text", ""),
                                    "Prediction": item.get("prediction", ""),
                                    "Confidence": item.get("confidence", 0),
                                    "Credibility Score": (1 - item.get("fake_probability", 0)) * 100,
                                    "Sensational Terms": item.get("features", {}).get("sensational_count", 0),
                                    "Credible Terms": item.get("features", {}).get("credible_count", 0),
                                    "Clickbait Phrases": item.get("features", {}).get("clickbait_count", 0)
                                }
                                for item in results
                            ])
                            
                            # Display results
                            st.subheader("Batch Processing Results")
                            st.dataframe(results_df)
                            
                            # Create summary chart
                            st.subheader("Prediction Summary")
                            
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
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Export results
                            st.subheader("Export Results")
                            
                            # Convert to CSV
                            csv_data = results_df.to_csv(index=False)
                            
                            # Download button
                            st.download_button(
                                label="Download Results CSV",
                                data=csv_data,
                                file_name="batch_analysis_results.csv",
                                mime="text/csv"
                            )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    elif page == "Comparison":
        st.title("Article Comparison")
        st.write("Compare multiple analyzed articles side by side.")
        
        display_comparison_view()
    
    elif page == "History":
        st.title("Analysis History")
        st.write("View your previous analyses.")
        
        display_history_view()

# Run the app
if __name__ == "__main__":
    main()