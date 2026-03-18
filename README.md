# 🛡️ NewsGuard: AI-Powered Fake News Detection System

**Version 1.2.0**

NewsGuard is a production-ready, full-stack application designed to combat misinformation using state-of-the-art Machine Learning. It provides a comprehensive suite for analyzing news credibility through contextual AI (BERT), statistical modeling (TF-IDF), and linguistic feature extraction.

---

## 🚀 Core Features

### 🧠 Hybrid Detection Engine
- **Ensemble Intelligence**: Combines **DistilBERT** (context-aware transformer) and **TF-IDF + Logistic Regression** (statistical feature weighting) for a balanced, high-confidence prediction.
- **Weighted Scoring**: Multi-model consensus with a 60/40 weighted split prioritizing BERT for deep semantic context.
- **Linguistic Heuristics**: Fallback analysis based on sensationalism, clickbait patterns, capitalization, and punctuation intensity.

### 🔍 Advanced Analysis Capabilities
- **Multi-Strategy Scraper**: Integrated `scraper.py` with redundant strategies (Article tags, CSS class pattern matching, and greedy paragraph collection) to extract clean content from any news URL.
- **Metadata Extraction**: Automatically fetches article author, source domain, publish date, and Open Graph descriptions.
- **Explainable AI (XAI)**: Provides human-readable summaries and factor breakdowns (🔴/🟢) explaining exactly why an article was flagged.
- **Side-by-Side Comparison**: Radar charts and metrics to compare features across multiple articles simultaneously.

### 📦 Enterprise-Grade Utilities
- **Batch Processing**: High-performance pipeline to analyze entire CSV/Excel datasets with progress tracking and exportable results.
- **Universal History**: SQLite-backed persistent storage of analysis history with full retrieval and deletion capabilities.
- **Rich Visualizations**: Gauge charts for trust levels, horizontal bars for feature intensity, and pie charts for keyword importance.

---

## 🛠️ Architecture & Tech Stack

The system follows a modular, service-oriented architecture:

- **Frontend**: `Streamlit` (Interactive Dashboard & Data Viz)
- **API Layer**: `Flask` (RESTful Backend with Rate Limiting)
- **ML Backbone**: `HuggingFace Transformers` (BERT), `Scikit-learn` (TF-IDF/LogReg)
- **Data Engine**: `BeautifulSoup4` (Scraping), `Pandas` (Processing), `SQLite` (History)
- **Deployment**: `Docker` & `Docker Compose`

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.9+
- Pip & Virtualenv

### 1. Manual Setup
```bash
# Clone and enter the directory
git clone https://github.com/GZ30eee/NewsGuard.git
cd NewsGuard

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python run.py
```
This single command orchestrates:
1.  **Data Initialization**: Downloads NLTK resources and sets up `data/` directories.
2.  **Service Launch**: Starts the Flask API on port `5000` and the Streamlit Dashboard on port `8501`.
3.  **Browser Auto-Open**: Automatically launches your default browser to the dashboard.

### 3. Docker Deployment
```bash
docker-compose up --build
```

---

## 📂 Project Structure

```text
NewsGuard/
├── app/
│   ├── models/          # BERT & TF-IDF implementation
│   ├── routes/          # API endpoints (predict, scrape, history)
│   ├── services/        # Prediction & ensemble logic
│   ├── utils/           # Scraper, Text Processor, Visualizer
│   ├── backend.py       # Flask server entry point
│   └── frontend.py      # Streamlit UI entry point
├── data/                # Dataset storage (fake_news.csv, real_news.csv)
├── models/              # Saved model weights and pickles
├── analysis_history/    # Persistent storage for past results
├── config.py            # Global settings (Thresholds, Paths, Ports)
├── run.py               # Main orchestrator (starts BE & FE)
├── train_models.py      # Automated training pipeline
└── requirements.txt     # Dependency list
```

---

## 🧪 Training Your Own Models

To retrain the system on your unique dataset:
1.  Place your data in `data/` as `fake_news.csv` and `real_news.csv`.
2.  Ensure CSVs have a `text` column.
3.  Run the training pipeline:
    ```bash
    python train_models.py
    ```

---

## 🔒 Security & Performance
- **Rate Limiting**: Integrated `Flask-Limiter` to prevent API abuse.
- **SSRF Protection**: URL scraping restricted to public domains with timeout safety.
- **Lazy Loading**: Transformation models are loaded into memory only when first called to optimize startup performance.

---

## 📜 License
Distibuted under the MIT License. See `LICENSE` for more information.

