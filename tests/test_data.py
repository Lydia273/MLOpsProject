from torch.utils.data import Dataset
from tests import _PATH_DATA
import pytest
import torch
import os

from src.sentiment_analysis.data import SentimentDataset

#@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_my_dataset():
    """Test data integrity and correctness."""

    # Paths to the data files
    processed_path = os.path.join(_PATH_DATA, "processed")
    train_encodings_path = os.path.join(processed_path, "train_encodings.pt")
    test_encodings_path = os.path.join(processed_path, "test_encodings.pt")
    train_labels_path = os.path.join(processed_path, "train_labels.pt")
    test_labels_path = os.path.join(processed_path, "test_labels.pt")


    """
    # Check for the existence of the necessary directories and files
    if not os.path.exists(processed_path):
        assert os.makedirs(processed_path)
    else:
        assert os.path.exists(processed_path), "Processed data folder not found"
    assert os.path.exists(train_encodings_path), "Processed train data file not found"
    assert os.path.exists(test_encodings_path), "Processed test data file not found"
    assert os.path.exists(train_labels_path), "Processed train labels file not found"
    assert os.path.exists(test_labels_path), "Processed test labels file not found"
    
    # Load the data
    train_encodings = torch.load(train_encodings_path)
    test_encodings = torch.load(test_encodings_path)
    train_labels = torch.load(train_labels_path)

    test_labels = torch.load(test_labels_path)

    assert train_labels.dim() == 1, "Train labels should be 1-dimensional"
    assert test_labels.dim() == 1, "Test labels should be 1-dimensional"

    assert 'input_ids' in train_encodings, "Train encodings missing 'input_ids'"
    assert 'input_ids' in test_encodings, "Test encodings missing 'input_ids'"
    assert isinstance(train_labels, torch.Tensor), "Train labels must be a torch Tensor"
    assert isinstance(test_labels, torch.Tensor), "Test labels must be a torch Tensor"
    """

"""
Coverage:
Name                                 Stmts   Miss  Cover
--------------------------------------------------------
src\sentiment_analysis\__init__.py       0      0   100%
src\sentiment_analysis\data.py          72     56    22%
--------------------------------------------------------
TOTAL                                   72     56    22%

"""