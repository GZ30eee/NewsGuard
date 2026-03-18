# NewsGuard: AI-Powered Fake News Detection 🛡️

NewsGuard is a comprehensive, production-ready application that utilizes state-of-the-art Machine Learning models to analyze news articles and determine their credibility.

## 🚀 Features

- **Multi-Model Inference**: Utilize **BERT** (Deep Learning) or **TF-IDF + Logistic Regression** (Broad Feature Analysis).
- **Ensemble Engine**: Combines multiple models for a weighted, highly accurate prediction.
- **URL Analysis**: Automatically scrape and analyze content from news URLs.
- **Batch Processing**: Upload CSV/Excel files for bulk analysis with progress tracking.
- **Rich Visualizations**: Credibility gauges, word clouds, feature intensity charts, and keyword importance.
- **Historical Analysis**: Track and manage previous analyses with integrated history and comparison views.
- **Modular & Secure Architecture**: Split across clean API routes, business services, and robust model wrappers.
- **Performance Optimized**: Features lazy model loading and request rate limiting.

## 🛠️ Tech Stack

- **Backend**: Python, Flask, Flask-Limiter, NLTK
- **ML Models**: HuggingFace Transformers (DistilBERT), Scikit-Learn (TF-IDF)
- **Frontend**: Streamlit, Plotly, CSS-in-Streamlit
- **DevOps**: Docker, Docker Compose

## 📦 Installation

### 1. Traditional Setup (Virtual Env)
```bash
# Clone the repository
cd NewsGuard

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python run.py
```
This will automatically:
1. Initialize directory structures.
2. Load/Train sample models if missing.
3. Start the Flask Backend (Port 5000).
4. Start the Streamlit Frontend (Port 8501).

## 🐳 Docker Deployment
```bash
docker-compose up --build
```

## 🧪 Model Training
To retrain your models with custom data:
1. Place your CSVs in `data/` (`fake_news.csv` and `real_news.csv`).
2. Run:
```bash
python train_models.py
```

## 🔒 Security
- **SSRF Protection**: URL scraping is restricted to public domains.
- **Input Sanitization**: All text is cleaned and length-limited to prevent DoS.
- **Rate Limiting**: Backend API is protected via `Flask-Limiter`.

## 📜 License
MIT License.
