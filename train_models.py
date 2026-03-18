import pandas as pd
import numpy as np
import logging
import os
import shutil
import time
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import config
from app.utils.text_processor import preprocess_text

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_data():
    """Load data from multiple sources in the data directory"""
    data_frames = []
    
    # Standard paths from config
    paths = [(config.FAKE_NEWS_PATH, 1), (config.REAL_NEWS_PATH, 0)]
    
    for path, label in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'text' in df.columns:
                df = df[['text']].copy()
                df['label'] = label
                data_frames.append(df)
    
    if not data_frames:
        raise ValueError("No training data found in data/ directory.")
    
    return pd.concat(data_frames, ignore_index=True).dropna()

def train_tfidf(data):
    from app.models.tfidf_model import TFIDFModel
    logger.info("Training TF-IDF model...")
    
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    
    # Preprocess
    X_train = [preprocess_text(t) for t in tqdm(X_train, desc="Preprocessing Train")]
    
    model = TFIDFModel()
    acc = model.train(X_train, y_train.values)
    
    # Versioning
    version_path = os.path.join(config.MODELS_DIR, f"tfidf_{int(time.time())}.pkl")
    model.save_model(version_path)
    logger.info(f"TF-IDF trained. Acc: {acc:.2%}. Version saved to {version_path}")

def train_bert(data, epochs=3, batch_size=8):
    logger.info("Starting BERT fine-tuning...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(config.BERT_MODEL_NAME, num_labels=2)
    model.to(device)
    
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.1, random_state=42)
    
    train_set = NewsDataset(X_train.values, y_train.values, tokenizer, config.MAX_LENGTH)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    
    model.train()
    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    # Save as main and version
    version_dir = os.path.join(config.MODELS_DIR, f"bert_{int(time.time())}")
    model.save_pretrained(config.BERT_MODEL_DIR)
    tokenizer.save_pretrained(config.BERT_MODEL_DIR)
    model.save_pretrained(version_dir)
    tokenizer.save_pretrained(version_dir)
    logger.info(f"BERT training complete. Saved to {config.BERT_MODEL_DIR} and {version_dir}")

if __name__ == "__main__":
    try:
        data = load_data()
        logger.info(f"Dataset loaded: {len(data)} samples")
        
        train_tfidf(data)
        
        # BERT training is optional/configurable based on hardware
        if torch.cuda.is_available() or len(data) < 1000:
            train_bert(data)
        else:
            logger.warning("Skipping BERT training to save resources (no CUDA). Use train_bert(data) manually if needed.")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)