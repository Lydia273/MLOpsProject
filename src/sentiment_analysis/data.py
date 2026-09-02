import os
import torch
import pandas as pd
from transformers import AlbertTokenizer
from torch.utils.data import Dataset
import typer
from loguru import logger

# Initialize Typer app
app = typer.Typer(help="CLI for preprocessing sentiment analysis data.")

@app.command()
def preprocess_data(
    raw_dir: str = typer.Argument(..., help="Directory containing a single CSV file (Train.csv)."),
    processed_dir: str = typer.Argument(..., help="Directory to save processed PyTorch tensors."),
    model_name: str = typer.Argument(..., help="Pre-trained model name for tokenizer."),
    max_length: int = typer.Option(128, help="Maximum sequence length for tokenization.")
) -> None:
    """
    Preprocess a single CSV (Train.csv) for sentiment analysis. Splits data into
    an 80/20 train/test split, then saves the resulting PyTorch tensors.

    Args:
        raw_dir (str): Path to the data directory containing Train.csv.
        processed_dir (str): Path to the directory where processed data will be saved.
        model_name (str): Name of the pre-trained model for tokenization.
        max_length (int, optional): Maximum length for tokenization. Defaults to 128.

    Example usage:
        python src/sentiment_analysis/data.py data/raw data/processed "albert-base-v2" --max-length 128
    """
    logger.info("Starting data preprocessing...")

    # Ensure the processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    logger.info(f"Processed directory '{processed_dir}' is ready.")

    # Define path to the single CSV file
    data_path = os.path.join(raw_dir, "Train.csv")

    # Verify that the CSV file exists
    if not os.path.isfile(data_path):
        logger.error(f"Train.csv not found in {raw_dir}.")
        raise FileNotFoundError(f"{data_path} does not exist.")

    # Load the dataset
    df = pd.read_csv(data_path)
    logger.info("Data loaded successfully.")

    # Handle missing values in 'Body' by replacing with empty string
    df['Body'] = df['Body'].fillna("")

    # Check if 'Sentiment Type' column exists
    if 'Sentiment Type' not in df.columns:
        logger.error("Train.csv must contain a 'Sentiment Type' column.")
        raise ValueError("Missing 'Sentiment Type' column in Train.csv.")

    # Encode sentiment labels using pandas factorize
    df['Sentiment_Label'], unique_labels = pd.factorize(df['Sentiment Type'])
    logger.info("Sentiment labels encoded.")

    # Save the unique labels for future reference (e.g., in evaluation or inference)
    torch.save(unique_labels, os.path.join(processed_dir, "unique_labels.pt"))
    logger.info("Unique labels saved.")

    # Shuffle the data to ensure a random distribution
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info("Data shuffled.")

    # Split index for 80/20 train/test
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    logger.info("Data split into train and test sets.")

    # Initialize the tokenizer
    try:
        tokenizer = AlbertTokenizer.from_pretrained(model_name)
        logger.info(f"Tokenizer loaded with model name '{model_name}'.")
    except Exception as e:
        logger.error(f"Failed to load tokenizer with model name '{model_name}': {e}")
        raise

    # Tokenize the training data
    train_encodings = tokenizer(
        train_df['Body'].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    logger.info("Training data tokenized.")

    # Tokenize the test data
    test_encodings = tokenizer(
        test_df['Body'].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    logger.info("Test data tokenized.")

    # Extract labels for training and testing data
    train_labels = torch.tensor(train_df['Sentiment_Label'].values)
    test_labels = torch.tensor(test_df['Sentiment_Label'].values)


    # Save tokenized inputs and labels
    torch.save(train_encodings, os.path.join(processed_dir, "train_encodings.pt"))
    torch.save(train_labels, os.path.join(processed_dir, "train_labels.pt"))
    torch.save(test_encodings, os.path.join(processed_dir, "test_encodings.pt"))
    torch.save(test_labels, os.path.join(processed_dir, "test_labels.pt"))
    logger.info("Tokenized data and labels saved successfully.")

    logger.info("Data preprocessing complete.")

class SentimentDataset(Dataset):
    """
    PyTorch Dataset for Sentiment Analysis.
    """

    def __init__(self, encodings, labels=None):
        """
        Initialize the dataset with encodings and optional labels.

        Args:
            encodings (dict): Tokenized inputs.
            labels (torch.Tensor, optional): Labels for the data. Defaults to None.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Retrieve the item at the specified index.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels (if available).
        """
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        """
        Return the total number of items in the dataset.

        Returns:
            int: Number of items.
        """
        return len(self.encodings['input_ids'])

def load_datasets(processed_dir: str) -> tuple[Dataset, Dataset]:
    """
    Load the processed datasets and create PyTorch Dataset objects.

    Args:
        processed_dir (str): Directory containing processed tensors.

    Returns:
        tuple[Dataset, Dataset]: Training and testing datasets.
    """
    # Load tokenized data and labels
    train_encodings = torch.load(os.path.join(processed_dir, "train_encodings.pt"))
    train_labels = torch.load(os.path.join(processed_dir, "train_labels.pt"))
    test_encodings = torch.load(os.path.join(processed_dir, "test_encodings.pt"))
    test_labels = torch.load(os.path.join(processed_dir, "test_labels.pt"))

    # Create Dataset objects
    train_dataset = SentimentDataset(train_encodings, train_labels)
    test_dataset = SentimentDataset(test_encodings, test_labels)

    return train_dataset, test_dataset

if __name__ == "__main__":
    app()
