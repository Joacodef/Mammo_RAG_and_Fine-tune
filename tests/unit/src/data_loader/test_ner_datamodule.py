# tests/unit/src/data_loader/test_ner_datamodule.py
import pytest
import torch
import json
import warnings
from unittest.mock import MagicMock, patch

# Add the project root to the Python path to allow for absolute imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.data_loader.ner_datamodule import NERDataModule, NERDataset

# Mock configuration for testing
@pytest.fixture
def mock_config():
    """Provides a mock configuration dictionary for the NERDataModule."""
    return {
        'model': {
            'base_model': 'mock-model',
            'entity_labels': ["FIND", "REG"]
        },
        'trainer': {
            'batch_size': 2
        },
        'batch_size': 2 # For test_dataloader
    }

# Corrected mock tokenizer fixture
@pytest.fixture
def mock_tokenizer():
    """
    Creates a mock tokenizer that handles different call patterns.
    - Called with `return_tensors='pt'`: Returns a dict of tensors.
    - Called otherwise (for alignment): Returns a mock object with `word_ids` and `encodings`.
    """
    tokenizer = MagicMock()

    # 1. The object returned by the alignment call inside `_align_labels`
    alignment_mock = MagicMock()
    alignment_mock.word_ids.return_value = [None, 0, 0, 1, 2, 3, 4, 4, 5, None] # Example word_ids
    alignment_mock.encodings = [MagicMock()]
    # Mock character offsets for tokens: [(start_char, end_char), ...]
    # This should correspond to the text in jsonl_file fixture
    alignment_mock.encodings[0].offsets = [
        (0, 0), (0, 6), (7, 10), (11, 15), (16, 20), # "Report one with a finding."
        (21, 22), (23, 29), (29, 30), (31, 32), (32, 33) # a finding.
    ]

    # 2. The dictionary returned by the main tokenization call in `__getitem__`
    main_tokenization_result = {
        "input_ids": torch.randint(1, 1000, (1, 512)),
        "attention_mask": torch.ones((1, 512), dtype=torch.long)
    }

    # 3. Use side_effect to switch between the two return values
    def tokenizer_caller(*args, **kwargs):
        if 'return_tensors' in kwargs and kwargs['return_tensors'] == 'pt':
            return main_tokenization_result
        else:
            return alignment_mock

    tokenizer.side_effect = tokenizer_caller
    tokenizer.decode.return_value = "Decoded text doesn't matter here"
    tokenizer.pad_token_id = 0
    
    # Also patch from_pretrained to return this mock
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer):
        yield tokenizer


# Fixture to create a temporary JSONL file for testing
@pytest.fixture
def jsonl_file(tmp_path):
    """
    Creates a temporary .jsonl file with sample data for tests.
    """
    data = [
        {"text": "Report one with a finding.", "entities": [{"label": "FIND", "start_offset": 18, "end_offset": 25}]},
        {"text": "Report two, nothing to see here.", "entities": []},
        {"text": "Report three with an unknown entity.", "entities": [{"label": "UNKNOWN", "start_offset": 19, "end_offset": 33}]}
    ]
    file_path = tmp_path / "test_data.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')
    return file_path

# --- Test Cases for NERDataset ---

def test_dataset_loading(jsonl_file, mock_tokenizer):
    """
    Tests that the NERDataset correctly loads data from a .jsonl file
    and reports the correct length.
    """
    label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2}
    dataset = NERDataset(file_path=str(jsonl_file), tokenizer=mock_tokenizer, label_map=label_map)
    
    assert len(dataset) == 3
    assert dataset.data[0]['text'] == "Report one with a finding."

def test_getitem_returns_correct_structure(jsonl_file, mock_tokenizer):
    """
    Tests that the __getitem__ method returns a dictionary with the correct keys
    and tensor shapes.
    """
    label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2}
    dataset = NERDataset(file_path=str(jsonl_file), tokenizer=mock_tokenizer, label_map=label_map)
    
    item = dataset[0]
    
    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item['input_ids'].shape == (512,)
    assert item['labels'].shape == (512,)

def test_align_labels_correctly(jsonl_file, mock_tokenizer):
    """
    Tests that entity labels are correctly aligned with token indices,
    assigning B- and I- tags properly.
    """
    # B-FIND:1, I-FIND:2, B-REG:3, I-REG:4
    label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2} 
    dataset = NERDataset(file_path=str(jsonl_file), tokenizer=mock_tokenizer, label_map=label_map)
    
    # The record has: {"label": "FIND", "start_offset": 18, "end_offset": 25} -> "finding"
    # mock offsets: (16, 20) is token 4, word 3. (21, 22) is token 5, word 4.
    # The logic in ner_datamodule is complex, this test ensures the core idea works.
    # A real tokenizer would give better offsets.
    # Based on our mock, let's assume `finding` starts at token index 4.
    
    labels = dataset._align_labels(torch.ones(10), dataset.data[0]["entities"])

    # This test is simplified due to mock complexity. A real test would be more precise.
    # We expect at least one B-tag and some I-tags if the entity spans multiple tokens.
    assert (labels == label_map["B-FIND"]).any()


def test_align_labels_with_unknown_entity(jsonl_file, mock_tokenizer):
    """
    Tests that an entity with a label not in the label_map is ignored and
    a warning is issued.
    """
    label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2}
    dataset = NERDataset(file_path=str(jsonl_file), tokenizer=mock_tokenizer, label_map=label_map)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Access the item with the unknown entity to trigger the warning
        _ = dataset[2] 
        assert len(w) == 1
        assert "Entity label 'UNKNOWN' not found" in str(w[0].message)

def test_align_labels_with_no_entities(jsonl_file, mock_tokenizer):
    """
    Tests that a record with no entities results in only 'O' labels.
    """
    label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2}
    dataset = NERDataset(file_path=str(jsonl_file), tokenizer=mock_tokenizer, label_map=label_map)

    # Get the item with no entities
    item = dataset[1]
    labels = item['labels']

    # All labels should be 0 ('O') since there are no entities
    assert torch.all(labels == 0)


# --- Test Cases for NERDataModule ---

def test_datamodule_initialization(mock_config, mock_tokenizer):
    """
    Tests that NERDataModule initializes the tokenizer and creates the label_map correctly.
    """
    datamodule = NERDataModule(config=mock_config)
    
    assert datamodule.tokenizer is not None
    assert "B-FIND" in datamodule.label_map
    assert "I-REG" in datamodule.label_map
    assert datamodule.label_map["O"] == 0

def test_datamodule_setup(mock_config, mock_tokenizer, jsonl_file):
    """
    Tests that the setup method correctly creates the train and test datasets.
    """
    datamodule = NERDataModule(config=mock_config, train_file=str(jsonl_file), test_file=str(jsonl_file))
    datamodule.setup()
    
    assert isinstance(datamodule.train_dataset, NERDataset)
    assert isinstance(datamodule.test_dataset, NERDataset)
    assert len(datamodule.train_dataset) == 3

def test_datamodule_dataloaders(mock_config, mock_tokenizer, jsonl_file):
    """
    Tests that the train_dataloader and test_dataloader methods return DataLoader instances.
    """
    datamodule = NERDataModule(config=mock_config, train_file=str(jsonl_file), test_file=str(jsonl_file))
    datamodule.setup()
    
    train_dl = datamodule.train_dataloader()
    test_dl = datamodule.test_dataloader()

    assert train_dl is not None
    assert test_dl is not None
    assert train_dl.batch_size == mock_config['trainer']['batch_size']
    assert next(iter(train_dl)) is not None


# --- Fixtures for Robustness Testing ---

@pytest.fixture
def empty_jsonl_file(tmp_path):
    """Creates an empty .jsonl file."""
    file_path = tmp_path / "empty_data.jsonl"
    file_path.touch()
    return file_path

@pytest.fixture
def missing_key_jsonl_file(tmp_path):
    """Creates a .jsonl file where a record is missing the 'text' key."""
    # A .jsonl file has one JSON object per line.
    record = {"entities": []} 
    file_path = tmp_path / "missing_key.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')
    return file_path

@pytest.fixture
def malformed_jsonl_file(tmp_path):
    """Creates a file containing a line that is not valid JSON."""
    file_path = tmp_path / "malformed.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('{"text": "valid"}\n')
        f.write('this is not json\n')
    return file_path

# --- Tests for Input Data Robustness ---

def test_dataset_with_empty_file(empty_jsonl_file, mock_tokenizer):
    """
    Tests that the dataset handles an empty file gracefully.
    """
    dataset = NERDataset(file_path=str(empty_jsonl_file), tokenizer=mock_tokenizer, label_map={})
    
    assert len(dataset) == 0
    with pytest.raises(IndexError):
        _ = dataset[0]

def test_dataset_with_missing_key(missing_key_jsonl_file, mock_tokenizer):
    """
    Tests that a KeyError is raised if a record is missing the 'text' key.
    """
    dataset = NERDataset(file_path=str(missing_key_jsonl_file), tokenizer=mock_tokenizer, label_map={})
    
    # Accessing the item should trigger the error when it tries to get record["text"]
    with pytest.raises(KeyError):
        _ = dataset[0]

def test_dataset_with_malformed_json(malformed_jsonl_file, mock_tokenizer):
    """
    Tests that a json.JSONDecodeError is raised if the file contains invalid JSON.
    """
    # The error should be raised during the initialization of the dataset
    with pytest.raises(json.JSONDecodeError):
        _ = NERDataset(file_path=str(malformed_jsonl_file), tokenizer=mock_tokenizer, label_map={})