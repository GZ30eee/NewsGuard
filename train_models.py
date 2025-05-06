import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import logging
import os
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import config
from app.models.bert_model import BERTModel
from app.models.tfidf_model import TFIDFModel
from app.utils.text_processor import preprocess_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NewsDataset(Dataset):
    """Custom Dataset for loading news data for BERT training"""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(fake_path, real_path):
    """
    Load and combine fake and real news datasets
    
    Args:
        fake_path (str): Path to fake news CSV
        real_path (str): Path to real news CSV
        
    Returns:
        pandas.DataFrame: Combined dataset
    """
    try:
        logger.info(f"Loading data from {fake_path} and {real_path}")
        
        # Load datasets
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)
        
        # Ensure required columns exist
        if 'text' not in fake_df.columns or 'text' not in real_df.columns:
            raise ValueError("Datasets must contain a 'text' column")
            
        # Add labels (1 for fake, 0 for real)
        fake_df['label'] = 1
        real_df['label'] = 0
        
        # Combine datasets
        data = pd.concat([fake_df[['text', 'label']], real_df[['text', 'label']]], ignore_index=True)
        
        # Drop any rows with missing text
        data = data.dropna(subset=['text'])
        
        logger.info(f"Loaded {len(data)} total samples")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def train_tfidf_model(data, model, test_size=0.2, random_state=42):
    """
    Train the TF-IDF model
    
    Args:
        data (pandas.DataFrame): Dataset with 'text' and 'label' columns
        model (TFIDFModel): TFIDFModel instance
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Training results including accuracy and classification report
    """
    try:
        logger.info("Starting TF-IDF model training")
        
        # Split data
        X = data['text'].values
        y = data['label'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Preprocess texts
        X_train_processed = [preprocess_text(text) for text in X_train]
        X_test_processed = [preprocess_text(text) for text in X_test]
        
        # Train model
        accuracy = model.train(X_train_processed, y_train)
        
        # Evaluate on test set
        y_pred = [model.predict(text)['prediction'] == 'FAKE' for text in X_test_processed]
        test_accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'], output_dict=True)
        
        results = {
            'train_accuracy': accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': report
        }
        
        logger.info(f"TF-IDF model training completed. Test accuracy: {test_accuracy:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error training TF-IDF model: {str(e)}")
        raise

def train_bert_model(data, model, tokenizer, test_size=0.2, random_state=42, batch_size=16, epochs=3):
    """
    Train the BERT model
    
    Args:
        data (pandas.DataFrame): Dataset with 'text' and 'label' columns
        model (transformers.AutoModelForSequenceClassification): BERT model
        tokenizer (transformers.AutoTokenizer): BERT tokenizer
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        
    Returns:
        dict: Training results including accuracy and classification report
    """
    try:
        logger.info("Starting BERT model training")
        
        # Split data
        X = data['text'].values
        y = data['label'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Create datasets
        train_dataset = NewsDataset(X_train, y_train, tokenizer, config.MAX_LENGTH)
        test_dataset = NewsDataset(X_test, y_test, tokenizer, config.MAX_LENGTH)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
        
        test_accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, target_names=['Real', 'Fake'], output_dict=True)
        
        logger.info(f"BERT model training completed. Test accuracy: {test_accuracy:.4f}")
        
        return {
            'test_accuracy': test_accuracy,
            'classification_report': report
        }
        
    except Exception as e:
        logger.error(f"Error training BERT model: {str(e)}")
        raise

def main():
    """Main function to load data and train models"""
    try:
        # Ensure model directories exist
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        
        # Clear BERT model directory if it exists
        if os.path.exists(config.BERT_MODEL_DIR):
            logger.info(f"Removing existing BERT model directory: {config.BERT_MODEL_DIR}")
            shutil.rmtree(config.BERT_MODEL_DIR)
        os.makedirs(config.BERT_MODEL_DIR, exist_ok=True)
        
        # Load data
        data = load_data(config.FAKE_NEWS_PATH, config.REAL_NEWS_PATH)
        
        # Initialize TF-IDF model
        logger.info("Initializing TF-IDF model")
        tfidf_model = TFIDFModel()
        
        # Initialize BERT model with pre-trained weights
        logger.info(f"Initializing BERT model with pre-trained {config.BERT_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.BERT_MODEL_NAME, 
            num_labels=2
        )
        
        # Train TF-IDF model
        tfidf_results = train_tfidf_model(data, tfidf_model)
        logger.info("TF-IDF Model Results:")
        logger.info(f"Train Accuracy: {tfidf_results['train_accuracy']:.4f}")
        logger.info(f"Test Accuracy: {tfidf_results['test_accuracy']:.4f}")
        logger.info(f"Classification Report:\n{pd.DataFrame(tfidf_results['classification_report']).transpose().to_string()}")
        
        # Save TF-IDF model
        tfidf_model.save_model(config.TFIDF_MODEL_PATH)
        
        # Train BERT model
        bert_results = train_bert_model(data, model, tokenizer)
        logger.info("BERT Model Results:")
        logger.info(f"Test Accuracy: {bert_results['test_accuracy']:.4f}")
        logger.info(f"Classification Report:\n{pd.DataFrame(bert_results['classification_report']).transpose().to_string()}")
        
        # Save BERT model
        logger.info(f"Saving BERT model to {config.BERT_MODEL_DIR}")
        model.save_pretrained(config.BERT_MODEL_DIR)
        tokenizer.save_pretrained(config.BERT_MODEL_DIR)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()