import streamlit as st
import pandas as pd
import requests
import json
import time
import os
import plotly.express as px
import plotly.graph_objects as go
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoint (from environment or default)
API_URL = os.getenv("API_URL", "http://localhost:5000/api")

# Set page configuration
st.set_page_config(
    page_title="NewsGuard | AI Fake News Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
if 'history' not in st.session_state: st.session_state.history = []
if 'comparison_items' not in st.session_state: st.session_state.comparison_items = []
if 'dark_mode' not in st.session_state: st.session_state.dark_mode = False
if 'last_result' not in st.session_state: st.session_state.last_result = None

# --- UI Styling ---
def apply_custom_styles():
    st.markdown("""
    <style>
        .stApp { background-color: white !important; color: #0e1117 !important; }
        [data-testid="stSidebar"] { background-color: #f8f9fa !important; }
        .result-card {
            padding: 2rem;
            border-radius: 1rem;
            margin-bottom: 1.5rem;
            background-color: #f0f2f6;
            border: 1px solid #d1d5db;
        }
        .metric-card {
            text-align: center;
            padding: 1rem;
            background: rgba(0, 0, 0, 0.05);
            border-radius: 0.5rem;
        }
        .fake-text { color: #ff4b4b; font-weight: bold; }
        .real-text { color: #00d488; font-weight: bold; }
        h1, h2, h3, p, span, li { color: #0e1117 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- API Helpers ---
def api_request(method, endpoint, **kwargs):
    try:
        url = f"{API_URL}{endpoint}"
        response = requests.request(method, url, timeout=60, **kwargs)
        if response.status_code == 200:
            return response.json()
        else:
            error = response.json().get('error', 'Unknown Error')
            st.error(f"API Error: {error}")
            return None
    except Exception as e:
        st.error(f"Connection Failed: {str(e)}")
        return None

# --- Page Components ---
def sidebar_nav():
    with st.sidebar:
        st.title("🛡️ NewsGuard")
        
        # Theme information (Removed toggle)
        st.caption("Standard Light Theme Active")
        
        st.divider()
        
        # Model Selection
        st.subheader("Compute Engine")
        model = st.radio(
            "Detecting Model",
            ["Ensemble", "BERT", "TF-IDF"],
            help="Ensemble combines both models for best results. BERT is contextual. TF-IDF is fast."
        )
        
        st.divider()
        
        # Navigation
        nav = st.radio(
            "Navigate",
            ["Analysis", "Batch Processing", "History", "Comparison"],
            index=0
        )
        
        st.divider()
        st.caption(f"Backend: {API_URL}")
        return nav, model.lower()

def show_analysis_page(model_type):
    st.header("🔍 News Credibility Analysis")
    
    # Input options
    mode = st.tabs(["Text Input", "URL Analysis"])
    
    with mode[0]:
        input_text = st.text_area(
            "Article Text", 
            height=300, 
            placeholder="Paste the news content here...",
            help="Longer articles provide more reliable analysis."
        )
        if st.button("Analyze Article", type="primary"):
            if input_text:
                result = api_request("POST", "/predict", json={"text": input_text, "model": model_type})
                if result:
                    st.session_state.last_result = result
                    st.rerun()
            else:
                st.warning("Please enter some text.")

    with mode[1]:
        url = st.text_input("Article URL", placeholder="https://news-site.com/article...")
        if st.button("Fetch & Analyze"):
            if url:
                scraped = api_request("POST", "/scrape", json={"url": url})
                if scraped:
                    result = api_request("POST", "/predict", json={"text": scraped['text'], "model": model_type})
                    if result:
                        st.session_state.last_result = result
                        st.rerun()
            else:
                st.warning("Enter a valid URL.")

    # Show Results if exist
    if st.session_state.last_result:
        display_results(st.session_state.last_result)

def display_results(res):
    st.divider()
    cols = st.columns([1, 1, 1])
    
    pred = res['prediction']
    color = "red" if pred == "FAKE" else "green" if pred == "REAL" else "orange"
    
    with cols[0]:
        st.markdown(f"### Result: :{color}[{pred}]")
        st.progress(res['fake_probability'], text=f"Fake Probability: {res['fake_probability']:.1%}")
        st.metric("Confidence", f"{res['confidence']:.1%}")
        
        if st.button("➕ Add to Comparison"):
            st.session_state.comparison_items.append(res)
            st.toast("Added to comparison!")

    with cols[1]:
        st.subheader("Explanation")
        st.write(res['explanation']['summary'])
        for factor in res['explanation']['factors']:
            icon = "🔴" if factor['type'] == 'negative' else "🟢"
            st.markdown(f"{icon} {factor['description']}")

    with cols[2]:
        st.subheader("Recommendations")
        for rec in res['explanation']['recommendations']:
            st.markdown(f"- {rec}")

    st.divider()
    # Visualizations
    vcols = st.columns(2)
    with vcols[0]:
        st.subheader("Feature Intensity")
        feats = res['features']
        fig = px.bar(
            x=[feats[k] for k in feats if 'count' in k],
            y=[k.replace('_count', '').title() for k in feats if 'count' in k],
            orientation='h',
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with vcols[1]:
        st.subheader("Important Keywords")
        words = res['important_words']
        if words:
            fig = px.pie(
                names=[w['word'] for w in words],
                values=[w['importance'] for w in words],
                hole=0.4,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_batch_page(model_type):
    st.header("📦 Batch Processing")
    uploaded_file = st.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        if 'text' not in df.columns:
            st.error("File must contain a 'text' column.")
            return

        titles = df['title'].tolist() if 'title' in df.columns else [f"Doc {i+1}" for i in range(len(df))]
        articles = [{"text": t, "title": title} for t, title in zip(df['text'], titles)]
        
        if st.button("Start Batch Analysis"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Use chunks for large batches
            chunk_size = 5
            all_results = []
            for i in range(0, len(articles), chunk_size):
                chunk = articles[i:i+chunk_size]
                status_text.text(f"Processing {i} to {i+len(chunk)}...")
                res = api_request("POST", "/batch-predict", json={"articles": chunk, "model": model_type})
                if res:
                    all_results.extend(res['results'])
                progress_bar.progress((i + len(chunk)) / len(articles))
            
            status_text.text("Processing Complete!")
            
            # Show summary
            res_df = pd.DataFrame([{"Title": r['title'], "Prediction": r['prediction'], "Confidence": r['confidence']} for r in all_results])
            st.dataframe(res_df, use_container_width=True)
            
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "batch_results.csv", "text/csv")

def show_history_page():
    st.header("🕒 Analysis History")
    history = api_request("GET", "/history")
    
    if not history:
        st.info("No history found.")
        return

    for item in history:
        with st.expander(f"{item['timestamp'][:16]} - {item['prediction']} ({item['confidence']:.1%})"):
            st.write(item.get('text_preview', 'No preview'))
            c1, c2, c3 = st.columns(3)
            if c1.button("View Full", key=f"view_{item['id']}"):
                full = api_request("GET", f"/history/{item['id']}")
                if full:
                    st.session_state.last_result = full
                    st.toast("Result loaded from history!")
            
            if c2.button("Compare", key=f"comp_{item['id']}"):
                full = api_request("GET", f"/history/{item['id']}")
                if full: st.session_state.comparison_items.append(full)
            
            if c3.button("🗑️ Delete", key=f"del_{item['id']}"):
                if api_request("DELETE", f"/history/{item['id']}"):
                    st.success("Deleted!")
                    st.rerun()

def show_comparison_page():
    st.header("⚖️ Side-by-Side Comparison")
    if not st.session_state.comparison_items:
        st.info("Add items from Analysis or History to compare.")
        return

    if st.button("🗑️ Clear All"):
        st.session_state.comparison_items = []
        st.rerun()

    # Create a nice comparison table
    items = st.session_state.comparison_items
    
    # Summary Metrics
    cols = st.columns(len(items))
    for i, item in enumerate(items):
        with cols[i]:
            card_color = "#ff4b4b" if item['prediction'] == "FAKE" else "#00d488"
            st.markdown(f"""
            <div style="border-left: 5px solid {card_color}; padding-left: 10px; margin-bottom: 20px;">
                <h4>Doc {i+1}</h4>
                <p>{item['prediction']}<br>Conf: {item['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
    # Radar Chart or Parallel coordinates for features
    st.subheader("Feature Comparison")
    feat_data = []
    for i, item in enumerate(items):
        f = item['features']
        row = {k.replace('_count',''): v for k,v in f.items() if 'count' in k}
        row['Doc'] = f"Doc {i+1}"
        feat_data.append(row)
    
    df_feat = pd.DataFrame(feat_data)
    fig = px.line_polar(df_feat.melt(id_vars='Doc'), r='value', theta='variable', color='Doc', line_close=True,
                       template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# --- Main App Execution ---
def main():
    apply_custom_styles()
    nav, model = sidebar_nav()
    
    if nav == "Analysis": show_analysis_page(model)
    elif nav == "Batch Processing": show_batch_page(model)
    elif nav == "History": show_history_page()
    elif nav == "Comparison": show_comparison_page()

if __name__ == "__main__":
    main()