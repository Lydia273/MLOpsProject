# train.py

import os
import torch
from torch.utils.data import DataLoader
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW
import torch.nn as nn
import typer
import hydra
from omegaconf import DictConfig
from data import load_datasets
from typing import Optional
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from loguru import logger
import wandb

logger.add("my_log.log", level="INFO", rotation="100 MB")

# Initialize Typer app
app = typer.Typer(help="CLI for training sentiment analysis model.")

@hydra.main(config_path="conf", config_name="config.yaml")
def train_model(cfg: DictConfig):
    # Initialize wandb
    wandb.init(
        project="sentiment-analysis",
        config={
            "model_name": cfg.hyperparameters.model_name,
            "batch_size": cfg.hyperparameters.batch_size,
            "learning_rate": cfg.hyperparameters.learning_rate,
            "epochs": cfg.hyperparameters.epochs,
            "num_labels": cfg.hyperparameters.num_labels
        }
    )

    # Access hyperparameters from wandb.config with fallback to cfg
    config = wandb.config
    model_name = config.get("model_name", cfg.hyperparameters.model_name)
    batch_size = config.get("batch_size", cfg.hyperparameters.batch_size)
    learning_rate = config.get("learning_rate", cfg.hyperparameters.learning_rate)
    epochs = config.get("epochs", cfg.hyperparameters.epochs)
    num_labels = config.get("num_labels", cfg.hyperparameters.num_labels)

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset, test_dataset = load_datasets(cfg.processed_dir)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=False)

    logger.info(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} testing samples.")

    # Initialize the model and tokenizer
    logger.info(f"Initializing model and tokenizer with {cfg.hyperparameters.model_name}...")
    tokenizer = AlbertTokenizer.from_pretrained(cfg.hyperparameters.model_name)
    model = AlbertForSequenceClassification.from_pretrained(cfg.hyperparameters.model_name, num_labels=cfg.hyperparameters.num_labels, ignore_mismatched_sizes=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    # Set up optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=cfg.hyperparameters.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    logger.info("Starting training...")

    # Training loop with profiling
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/train")) as prof:
        model.train()
        for epoch in range(cfg.hyperparameters.epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                prof.step()

                # Log batch metrics to wandb
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch": batch_idx + epoch * len(train_dataloader)
                })

                if batch_idx % 100 == 0:
                    logger.debug(f"Epoch {epoch + 1}, Batch {batch_idx}: Loss = {loss.item():.4f}")

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Log epoch metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
            })

    # Save the model if save_path is provided
    if cfg.hyperparameters.save_path:
        os.makedirs(cfg.hyperparameters.save_path, exist_ok=True)
        model_save_path = os.path.join(cfg.hyperparameters.save_path, "sentiment_model.pth")
        torch.save(model.state_dict(), model_save_path)
        tokenizer.save_pretrained(cfg.hyperparameters.save_path)
        logger.success(f"Model saved to {model_save_path}")
        
        # Log model as artifact
        artifact = wandb.Artifact('sentiment-model', type='model')
        artifact.add_file(model_save_path)
        wandb.log_artifact(artifact)

@app.command()
def train(
    processed_dir: str = typer.Argument(..., help="Path to directory containing processed data tensors."),
    config_name: str = typer.Option("config.yaml", help="Name of the configuration file."),
):
    """
    Train the sentiment model.
    """
    logger.info("Initializing training configuration...")
    hydra.initialize(config_path="conf")
    cfg = hydra.compose(config_name=config_name)
    cfg.processed_dir = processed_dir

    train_model(cfg)

@hydra.main(config_path="conf", config_name="config.yaml")
def evaluate_model(cfg: DictConfig):
    # Initialize wandb for evaluation run
    wandb.init(
        project="sentiment-analysis",
        config={
            "model_name": cfg.hyperparameters.model_name,
            "batch_size": cfg.hyperparameters.batch_size,
            "num_labels": cfg.hyperparameters.num_labels
        },
        job_type="evaluation"
    )

    # Load datasets
    logger.info("Loading test dataset...")
    _, test_dataset = load_datasets(cfg.processed_dir)
    logger.info(f"Loaded {len(test_dataset)} testing samples.")

    # Create data loader
    val_dataloader = DataLoader(test_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=False)

    # Initialize model and tokenizer
    logger.info("Initializing model and tokenizer...")
    tokenizer = AlbertTokenizer.from_pretrained(cfg.hyperparameters.model_name)
    model = AlbertForSequenceClassification.from_pretrained(cfg.hyperparameters.model_name, num_labels=cfg.hyperparameters.num_labels)

    # Load the saved model state dictionary from .pth format
    state_dict_path = os.path.join(cfg.model_path, "sentiment_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(state_dict_path, map_location=device))

    logger.info(f"Evaluating on {device}")
    model.to(device)

    # Evaluation loop with profiling
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/evaluate")) as prof:
        model.eval()
        total, correct = 0, 0
        predictions, true_labels = [], []

        logger.info("Starting evaluation...")
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
                labels = batch["labels"].to(device)
                outputs = model(**inputs)
                _, preds = torch.max(outputs.logits, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                prof.step()

        accuracy = correct / total
        logger.success(f"Evaluation complete. Accuracy: {accuracy * 100:.2f}%")
        
        # Log evaluation metrics to wandb
        wandb.log({
            "test_accuracy": accuracy,
            "total_samples": total
        })

@app.command()
def evaluate(
    processed_dir: str = typer.Argument(..., help="Path to directory containing processed data tensors."),
    config_name: str = typer.Option("config.yaml", help="Name of the configuration file."),
    model_path: str = typer.Argument(..., help="Path to the saved model."),
):
    """
    Evaluate the sentiment model.

    Command:
    python src/sentiment_analysis/train.py evaluate ./data/processed ./models/trained_sentiment_model --config-name config.yaml
    """
    logger.info("Initializing evaluation configuration...")
    hydra.initialize(config_path="conf")
    cfg = hydra.compose(config_name=config_name)
    cfg.processed_dir = processed_dir
    cfg.model_path = model_path
    evaluate_model(cfg)

if __name__ == "__main__":
    app()