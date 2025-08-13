# tests/integration/test_re_pipeline.py
import pytest
import yaml
from pathlib import Path
import json
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.training.run_re_trainig import run_batch_re_training
from scripts.evaluation.run_evaluation import run_evaluation

# --- Fixtures for RE Test Data and Configuration ---

@pytest.fixture
def re_integration_config():
    """
    Provides a minimal configuration for the RE integration test.
    """
    return {
        'seed': 42,
        'trainer': {
            'n_epochs': 1,
            'batch_size': 2, # RE can often use slightly larger batches
            'learning_rate': 2e-5,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'device': "cpu"
        },
        'model': {
            'base_model': 'prajjwal1/bert-tiny',
            'relation_labels': ["describir", "ubicar", "No_Relation"]
        },
        'paths': {
            'output_dir': "output/models_re"
        }
    }

@pytest.fixture
def re_training_data():
    """
    Provides a few records for the RE training set, including entities and relations.
    """
    return [
        {
            "text": "Nódulo periareolar derecho bien delimitado.",
            "entities": [
                {"id": 1, "label": "HALL", "start_offset": 0, "end_offset": 6},
                {"id": 2, "label": "REG", "start_offset": 7, "end_offset": 26}
            ],
            "relations": [
                {"from_id": 1, "to_id": 2, "type": "ubicar"}
            ]
        },
        {
            "text": "Microcalcificaciones agrupadas.",
            "entities": [
                {"id": 3, "label": "HALL", "start_offset": 0, "end_offset": 20},
                {"id": 4, "label": "CARACT", "start_offset": 21, "end_offset": 31}
            ],
            "relations": [
                {"from_id": 3, "to_id": 4, "type": "describir"}
            ]
        }
    ]

@pytest.fixture
def re_test_data():
    """Provides a few records for the RE test/evaluation set."""
    return [
        {
            "text": "Nódulo en mama izquierda.",
            "entities": [
                {"id": 5, "label": "HALL", "start_offset": 0, "end_offset": 6},
                {"id": 6, "label": "REG", "start_offset": 10, "end_offset": 24}
            ],
            "relations": [
                {"from_id": 5, "to_id": 6, "type": "ubicar"}
            ]
        }
    ]

# --- RE Integration Test ---

def test_re_training_and_evaluation_pipeline(tmp_path, re_integration_config, re_training_data, re_test_data):
    """
    Tests the complete RE pipeline from training to evaluation.
    """
    # --- 1. Setup temporary directory and files ---
    partition_dir = tmp_path / "processed_re" / "train-2" / "sample-1"
    partition_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = partition_dir / "train.jsonl"
    with open(train_file, 'w') as f:
        for record in re_training_data:
            f.write(json.dumps(record) + '\n')
            
    test_file_path = tmp_path / "processed_re" / "test.jsonl"
    test_file_path.parent.mkdir(exist_ok=True)
    with open(test_file_path, 'w') as f:
        for record in re_test_data:
            f.write(json.dumps(record) + '\n')

    training_config_path = tmp_path / "training_re_config.yaml"
    re_integration_config['paths']['output_dir'] = str(tmp_path / "output" / "models_re")
    with open(training_config_path, 'w') as f:
        yaml.dump(re_integration_config, f)

    # --- 2. Run RE Training ---
    print("\n--- Running RE Training ---")
    run_batch_re_training(
        config_path=str(training_config_path),
        partition_dir=str(tmp_path / "processed_re" / "train-2")
    )

    # --- 3. Assert Training Outputs ---
    model_name = re_integration_config['model']['base_model'].replace("/", "_")
    expected_model_dir = Path(re_integration_config['paths']['output_dir']) / model_name / "train-2" / "sample-1"
    
    assert expected_model_dir.exists(), "RE Model output directory was not created."
    weights_file_exists = (
        (expected_model_dir / "pytorch_model.bin").exists() or
        (expected_model_dir / "model.safetensors").exists()
    )
    assert weights_file_exists, "RE Model weights file is missing."
    assert (expected_model_dir / "config.json").exists(), "RE Model config file is missing."
    print("--- RE Training Successful and Artifacts Verified ---")

    # --- 4. Run RE Evaluation ---
    print("\n--- Running RE Evaluation ---")
    evaluation_config = {
        'task': "re",
        'model_path': str(expected_model_dir),
        'test_file': str(test_file_path),
        'model': { 'relation_labels': re_integration_config['model']['relation_labels'] },
        'output_dir': str(tmp_path / "output" / "evaluation_results_re"),
        'batch_size': 1
    }
    
    # Call the evaluation function directly with the config dictionary
    report = run_evaluation(evaluation_config)

    # --- 5. Assert Evaluation Outputs ---
    # The function returns the report, which we can check directly
    assert "ubicar" in report, "'ubicar' relation not found in evaluation report."
    assert "accuracy" in report, "Accuracy not found in RE evaluation report."
    assert "auc" in report, "AUC score not found in RE evaluation report."
    
    # Verify that the individual metrics file was created
    output_dir = Path(evaluation_config['output_dir'])
    expected_metrics_file = output_dir / f"evaluation_metrics_{expected_model_dir.name}.json"
    assert expected_metrics_file.exists(), "Individual RE metrics file was not created."

    print("--- RE Evaluation Successful and Metrics Verified ---")