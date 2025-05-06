# NewsGuard

A comprehensive application for detecting fake news using machine learning.

## Features

- **Text Analysis**: Analyze news articles to determine if they might be fake or real
- **URL Analysis**: Fetch and analyze news articles from URLs
- **Batch Processing**: Analyze multiple articles at once via CSV upload
- **Multiple Models**: Choose between BERT (high accuracy) and TF-IDF (fast) models
- **Detailed Explanations**: Get insights into why an article might be fake or real
- **Visualizations**: View word clouds, feature analysis, and more
- **Comparison Tools**: Compare multiple articles side by side
- **History Tracking**: Review past analyses
- **Export Options**: Download results in JSON or CSV format

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download NLTK resources (if not automatically downloaded):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Usage

### Running the Application

Start the application with:

```
python run.py
```

This will launch both the backend API server and the Streamlit frontend, and open your web browser to the application.

## Models

The application uses two different models for fake news detection:

1. **BERT Model**: A transformer-based model that provides high accuracy but is slower to process.
2. **TF-IDF + Logistic Regression**: A classic machine learning approach that is faster but less accurate.

## Project Structure

```
fake-news-detector/
├── app/                  # Main application code
│   ├── frontend.py       # Streamlit UI
│   └── backend.py        # Flask API
├── data/                 # Training data
│   ├── fake_news.csv     # Sample fake news for training
│   └── real_news.csv     # Sample real news for training
├── models/               # Saved model files
├── analysis_history/     # Saved analysis results
├── requirements.txt      # Dependencies
├── README.md             # Documentation
└── run.py                # Main entry point
```

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed correctly
2. Check that both the backend and frontend servers are running
3. Look at the app.log file for error messages
4. Ensure you have an internet connection for URL scraping
5. Try restarting the application

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```

## How to Run the Application

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python run.py
   ```
   
3. The application will start both the backend and frontend servers and open your web browser to the Streamlit interface.
