from transformers import AlbertForSequenceClassification, AlbertTokenizer
import torch
import torch.nn as nn
from loguru import logger

class SentimentModel(nn.Module):
    """
    Sentiment analysis model using ALBERT for 3-label classification: Negative, Neutral, Positive.
    """
    def __init__(self, model_name="albert-base-v2", num_labels=3):
        super().__init__()
        self.num_labels = num_labels
        self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        
        # Load ALBERT with a classification head
        self.model = AlbertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        logger.info(f"Model {model_name} initialized with {num_labels} labels.")

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model. If labels are provided, returns loss along with outputs.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        logger.debug("Forward pass completed.")
        return outputs

    def predict(self, text):
        """
        Predict sentiment probabilities for a single piece of text.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        logger.info(f"Prediction made for text: {text}")
        return probabilities


def train_model(model, train_dataloader, optimizer, criterion, device, num_epochs=5):
    """
    Function to train the model.
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


def evaluate_model(model, val_dataloader, device):
    """
    Function to evaluate the model.
    """
    model.eval()
    total, correct = 0, 0
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = correct / total
    logger.info(f"Validation Accuracy: {accuracy:.4f}")
    return predictions, true_labels


def save_model(model, tokenizer, save_path):
    """
    Save only the fine-tuned weights and tokenizer.
    """
    model.model.save_pretrained(save_path)  # Save model weights
    tokenizer.save_pretrained(save_path)   # Save tokenizer
    logger.info(f"Model and tokenizer saved to {save_path}")


def load_model(save_path):
    """
    Load the fine-tuned model and tokenizer from the saved path.
    """
    model = AlbertForSequenceClassification.from_pretrained(save_path)
    tokenizer = AlbertTokenizer.from_pretrained(save_path)
    logger.info(f"Model and tokenizer loaded from {save_path}")
    return model, tokenizer