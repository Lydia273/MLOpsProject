import pytest
import torch
from transformers import AlbertForSequenceClassification, AlbertTokenizer
from src.sentiment_analysis.model import SentimentModel, train_model, evaluate_model, save_model, load_model

# Fixture to initialize the model
@pytest.fixture
def model():
    return SentimentModel()

# Test model initialization
def test_model_initialization(model):
    assert isinstance(model.model, AlbertForSequenceClassification), "Model should be an instance of AlbertForSequenceClassification"
    assert model.num_labels == 3, "Number of labels should be 3"

# Test forward pass
def test_forward_pass(model):
    input_ids = torch.randint(0, 2000, (1, 10))
    attention_mask = torch.ones(1, 10)
    outputs = model.forward(input_ids, attention_mask)
    assert 'loss' in outputs or 'logits' in outputs, "Forward should return a dictionary with 'loss' or 'logits'"

# Test prediction
def test_prediction(model):
    text = "The product is great!"
    probabilities = model.predict(text)
    assert probabilities.shape == (1, model.num_labels), "Probabilities should have the shape (1, num_labels)"

# Mock data for training and validation
@pytest.fixture
def dataloader():
    class MockDataLoader:
        def __iter__(self):
            return iter([{
                'input_ids': torch.randint(0, 2000, (2, 10)),
                'attention_mask': torch.ones(2, 10),
                'labels': torch.tensor([0, 2])
            }])

        def __len__(self):
            return 1

    return MockDataLoader()

# Test training function
def test_train_model(model, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        train_model(model, dataloader, optimizer, criterion, device, num_epochs=1)
        assert True, "Training function should complete without error"
    except Exception as e:
        assert False, f"Training function failed with error {e}"

# Test evaluate function
def test_evaluate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions, true_labels = evaluate_model(model, dataloader, device)
    assert len(predictions) == len(true_labels), "Should return predictions for each label"

# Test save and load functionality
def test_save_load_model(tmp_path, model):
    tokenizer = model.tokenizer
    save_model(model, tokenizer, tmp_path)
    loaded_model, loaded_tokenizer = load_model(tmp_path)
    assert loaded_model is not None, "Loaded model should not be None"
    assert loaded_tokenizer is not None, "Loaded tokenizer should not be None"

"""
Coverage:

Name                                 Stmts   Miss  Cover
--------------------------------------------------------
src\sentiment_analysis\__init__.py       0      0   100%
src\sentiment_analysis\model.py         66      0   100%
--------------------------------------------------------
TOTAL                                   66      0   100%


"""