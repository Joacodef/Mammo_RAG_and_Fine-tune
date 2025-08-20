# tests/unit/scripts/evaluation/test_generate_finetuned_predictions.py
import pytest
import json
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.evaluation.generate_finetuned_predictions import run_prediction_and_save, convert_numpy_types

# --- Fixtures for Testing ---

@pytest.fixture
def ner_config(tmp_path):
    """Provides a minimal configuration for a NER prediction task."""
    return {
        'task': 'ner',
        'model_path': str(tmp_path / 'models' / 'ner_model'),
        'test_file': str(tmp_path / 'data' / 'test.jsonl'),
        'output_dir': str(tmp_path / 'output' / 'ner_results')
    }

@pytest.fixture
def re_config(tmp_path):
    """Provides a minimal configuration for a RE prediction task."""
    return {
        'task': 're',
        'model_path': str(tmp_path / 'models' / 're_model'),
        'test_file': str(tmp_path / 'data' / 'test.jsonl'),
        'output_dir': str(tmp_path / 'output' / 're_results'),
        'model': { # REDataModule requires relation_labels in config
            'relation_labels': ['ubicar', 'describir']
        }
    }

# --- Mocks for Core Dependencies ---
# We patch the specific modules where they are looked up in the script under test.
@patch('scripts.evaluation.generate_finetuned_predictions.REDataModule')
@patch('scripts.evaluation.generate_finetuned_predictions.NERDataModule')
@patch('scripts.evaluation.generate_finetuned_predictions.REModel')
@patch('scripts.evaluation.generate_finetuned_predictions.BertNerModel')
@patch('scripts.evaluation.generate_finetuned_predictions.Predictor')
# The newline character is removed from read_data to prevent a JSON decoding error.
@patch('builtins.open', new_callable=mock_open, read_data='{"text": "Sample text."}')
@patch('pathlib.Path.mkdir')
def test_ner_workflow(mock_mkdir, mock_file_open, mock_predictor, mock_bert_ner_model, mock_re_model, mock_ner_datamodule, mock_re_datamodule, ner_config):
    """
    Tests the end-to-end prediction generation workflow for a NER task.
    """
    # --- Setup Mocks ---
    mock_predictor_instance = mock_predictor.return_value
    mock_predictor_instance.predict.return_value = (
        [[1, 0]], # Mock predictions
        [[1, 0]], # Mock true labels
        np.array([]) # Mock logits (not used in this script)
    )

    # --- Act ---
    run_prediction_and_save(ner_config)

    # --- Assertions ---
    # 1. Verify correct modules were initialized for NER
    mock_ner_datamodule.assert_called_once_with(config=ner_config, test_file=ner_config['test_file'])
    mock_bert_ner_model.assert_called_once_with(base_model=ner_config['model_path'])
    assert not mock_re_datamodule.called
    assert not mock_re_model.called

    # 2. Verify Predictor was initialized and used
    mock_predictor.assert_called_once()
    mock_predictor_instance.predict.assert_called_once()

    # 3. Verify file I/O
    mock_file_open.assert_any_call(ner_config['test_file'], 'r', encoding='utf-8')
    
    # Check that the output file was written to
    handle = mock_file_open()
    written_data = handle.write.call_args[0][0]
    
    # 4. Verify output content
    expected_output = {
        "source_text": "Sample text.",
        "true_labels": [1, 0],
        "predicted_labels": [1, 0]
    }
    assert json.loads(written_data) == expected_output

@patch('scripts.evaluation.generate_finetuned_predictions.REDataModule')
@patch('scripts.evaluation.generate_finetuned_predictions.NERDataModule')
@patch('scripts.evaluation.generate_finetuned_predictions.REModel')
@patch('scripts.evaluation.generate_finetuned_predictions.BertNerModel')
@patch('scripts.evaluation.generate_finetuned_predictions.Predictor')
@patch('builtins.open', new_callable=mock_open, read_data='{"text": "Sample RE text."}')
@patch('pathlib.Path.mkdir')
def test_re_workflow(mock_mkdir, mock_file_open, mock_predictor, mock_bert_ner_model, mock_re_model, mock_ner_datamodule, mock_re_datamodule, re_config):
    """
    Tests the end-to-end prediction generation workflow for a RE task.
    """
    # --- Setup Mocks ---
    mock_predictor_instance = mock_predictor.return_value
    # For RE, predict returns a flat list of labels, one for each instance.
    mock_predictor_instance.predict.return_value = ([0], [1], np.array([]))

    # --- Act ---
    run_prediction_and_save(re_config)

    # --- Assertions ---
    # 1. Verify correct modules were initialized for RE
    mock_re_datamodule.assert_called_once_with(config=re_config, test_file=re_config['test_file'])
    mock_re_model.assert_called_once()
    assert not mock_ner_datamodule.called
    assert not mock_bert_ner_model.called
    
    # 2. Verify output content
    handle = mock_file_open()
    written_data = handle.write.call_args[0][0]
    # The expected labels should be integers, not lists, for the RE task.
    expected_output = {
        "source_text": "Sample RE text.",
        "true_labels": 1,
        "predicted_labels": 0
    }
    assert json.loads(written_data) == expected_output

def test_convert_numpy_types():
    """
    Tests the convert_numpy_types helper function to ensure it correctly
    converts numpy types to native Python types for JSON serialization.
    """
    numpy_data = {
        "integer": np.int64(10),
        "float": np.float32(3.14),
        "list_of_nums": [np.int32(1), np.float64(2.5)],
        "nested": {
            "array": np.array([1, 2, 3])
        }
    }
    
    converted_data = convert_numpy_types(numpy_data)
    
    # Assert types are converted to native Python types
    assert isinstance(converted_data["integer"], int)
    assert isinstance(converted_data["float"], float)
    assert isinstance(converted_data["list_of_nums"][0], int)
    assert isinstance(converted_data["list_of_nums"][1], float)
    assert isinstance(converted_data["nested"]["array"], list)
    
    # Assert values are preserved
    assert converted_data["nested"]["array"] == [1, 2, 3]

def test_invalid_task_raises_error(ner_config):
    """
    Tests that the function raises a ValueError if the task in the config
    is not 'ner' or 're'.
    """
    invalid_config = ner_config.copy()
    invalid_config['task'] = 'unknown_task'
    
    with pytest.raises(ValueError, match="Configuration file must specify a 'task': 'ner' or 're'"):
        run_prediction_and_save(invalid_config)