# tests/integration/test_ner_pipeline.py
import pytest
import yaml
from pathlib import Path
import json
import sys

# Add the project root to the Python path to allow for absolute imports
# This is crucial for the integration test to find the src and scripts modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.training.run_ner_training import run_batch_training
from scripts.evaluation.run_evaluation import run_evaluation

# --- Fixtures for Test Data and Configuration ---

@pytest.fixture
def ner_integration_config():
    """
    Provides a minimal, fast configuration for the NER integration test.
    This uses a very small model to speed up download and training time.
    """
    return {
        'seed': 42,
        'trainer': {
            'n_epochs': 1,
            'batch_size': 1,
            'learning_rate': 2e-5,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'device': "cpu" # Force CPU to ensure test runs on any machine
        },
        'model': {
            'base_model': 'prajjwal1/bert-tiny', # A very small model for fast testing
            'entity_labels': ["FIND", "REG"]
        },
        'paths': {
            'output_dir': "output/models"
        }
    }

@pytest.fixture
def ner_training_data():
    """Provides a few records for the training set."""
    return [
        {"text": "Report one has a finding in the REG region.", "entities": [{"label": "FIND", "start_offset": 19, "end_offset": 26}, {"label": "REG", "start_offset": 34, "end_offset": 44}]},
        {"text": "Report two has a REG.", "entities": [{"label": "REG", "start_offset": 17, "end_offset": 20}]}
    ]

@pytest.fixture
def ner_test_data():
    """Provides a few records for the test/evaluation set."""
    return [
        {"text": "This test has a FIND.", "entities": [{"label": "FIND", "start_offset": 17, "end_offset": 21}]},
        {"text": "And this one has a REG.", "entities": [{"label": "REG", "start_offset": 20, "end_offset": 23}]}
    ]

# --- Integration Test ---

def test_ner_training_and_evaluation_pipeline(tmp_path, ner_integration_config, ner_training_data, ner_test_data):
    """
    Tests the complete NER pipeline from training to evaluation.

    This test performs the following steps:
    1. Sets up a temporary directory structure with test data and configs.
    2. Runs the main training script on a minimal dataset.
    3. Asserts that the expected model artifacts were created.
    4. Runs the main evaluation script using the newly trained model.
    5. Asserts that the evaluation metrics file was created.
    """
    # --- 1. Setup temporary directory and files ---
    
    # Create training data partition directory
    partition_dir = tmp_path / "processed" / "train-2" / "sample-1"
    partition_dir.mkdir(parents=True, exist_ok=True)
    
    # Write training data
    train_file = partition_dir / "train.jsonl"
    with open(train_file, 'w') as f:
        for record in ner_training_data:
            f.write(json.dumps(record) + '\n')
            
    # Write test data
    test_file_path = tmp_path / "processed" / "test.jsonl"
    test_file_path.parent.mkdir(exist_ok=True)
    with open(test_file_path, 'w') as f:
        for record in ner_test_data:
            f.write(json.dumps(record) + '\n')

    # Write training config
    training_config_path = tmp_path / "training_ner_config.yaml"
    ner_integration_config['paths']['output_dir'] = str(tmp_path / "output" / "models")
    with open(training_config_path, 'w') as f:
        yaml.dump(ner_integration_config, f)

    # --- 2. Run Training ---
    print("\n--- Running NER Training ---")
    run_batch_training(
        config_path=str(training_config_path),
        partition_dir=str(tmp_path / "processed" / "train-2")
    )

    # --- 3. Assert Training Outputs ---
    model_name = ner_integration_config['model']['base_model'].replace("/", "_")
    expected_model_dir = Path(ner_integration_config['paths']['output_dir']) / model_name / "train-2" / "sample-1"
    
    assert expected_model_dir.exists(), "Model output directory was not created."
    
    # Check for either the standard PyTorch weights file or the SafeTensors equivalent
    weights_file_exists = (
        (expected_model_dir / "pytorch_model.bin").exists() or
        (expected_model_dir / "model.safetensors").exists()
    )
    assert weights_file_exists, "Model weights file (pytorch_model.bin or model.safetensors) is missing."
    
    assert (expected_model_dir / "config.json").exists(), "Model config file is missing."
    assert (expected_model_dir / "tokenizer_config.json").exists(), "Tokenizer config file is missing."
    print("--- Training Successful and Artifacts Verified ---")

    # --- 4. Run Evaluation ---
    print("\n--- Running NER Evaluation ---")
    
    # Prepare evaluation config
    evaluation_config = {
        'task': "ner",
        'model_path': str(expected_model_dir),
        'test_file': str(test_file_path),
        'entity_labels': ner_integration_config['model']['entity_labels'],
        'output_dir': str(tmp_path / "output" / "evaluation_results"),
        'batch_size': 1
    }
    evaluation_config_path = tmp_path / "evaluation_ner_config.yaml"
    with open(evaluation_config_path, 'w') as f:
        yaml.dump(evaluation_config, f)
        
    run_evaluation(config_path=str(evaluation_config_path))

    # --- 5. Assert Evaluation Outputs ---
    expected_metrics_file = Path(evaluation_config['output_dir']) / f"evaluation_metrics_{expected_model_dir.name}.json"
    assert expected_metrics_file.exists(), "Evaluation metrics file was not created."
    
    with open(expected_metrics_file, 'r') as f:
        metrics = json.load(f)
    
    assert "FIND" in metrics, "FIND entity not found in evaluation report."
    assert "REG" in metrics, "REG entity not found in evaluation report."
    # The seqeval report includes several averages but not a single 'accuracy' key.
    # We will check for 'weighted avg' as a reliable indicator of a valid report.
    assert "weighted avg" in metrics, "Weighted average not found in evaluation report."
    print("--- Evaluation Successful and Metrics Verified ---")