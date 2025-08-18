import pytest
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.evaluation.calculate_final_metrics import (
    calculate_finetuned_metrics,
    calculate_rag_metrics
)

# --- Fixtures for Testing ---

@pytest.fixture
def mock_ner_config():
    """Provides a mock NER configuration for testing the label map."""
    return {
        'model': {
            'entity_labels': ["FIND", "REG"]
        }
        # Other keys (trainer, paths) are not needed for this test
    }

@pytest.fixture
def mock_finetuned_predictions():
    """Provides mock raw prediction data from a fine-tuned model."""
    # Based on a label map where O=0, B-FIND=1, I-FIND=2, B-REG=3, I-REG=4
    return [
        {
            "source_text": "A finding in the region.",
            "true_labels": [1, 0, 0, 3, 0], # B-FIND, O, O, B-REG, O
            "predicted_labels": [1, 0, 0, 0, 0] # B-FIND, O, O, O, O
        },
        {
            "source_text": "Another finding.",
            "true_labels": [0, 1, 0], # O, B-FIND, O
            "predicted_labels": [0, 1, 0] # O, B-FIND, O
        }
    ]

@pytest.fixture
def mock_rag_predictions():
    """Provides mock raw prediction data from a RAG pipeline."""
    return [
        {
            "source_text": "Nódulo en mama derecha.",
            "true_entities": [
                {"start_offset": 0, "end_offset": 6, "label": "FIND"},
                {"start_offset": 10, "end_offset": 23, "label": "REG"}
            ],
            "predicted_entities": [
                {"text": "Nódulo", "label": "FIND"}, # True Positive
                {"text": "mama izquierda", "label": "REG"} # False Positive
            ]
            # This setup implies one FN for "mama derecha"
        }
    ]

# --- Test Cases ---

@patch('scripts.evaluation.calculate_final_metrics.NERDataModule')
def test_calculate_finetuned_metrics(mock_datamodule, mock_finetuned_predictions, mock_ner_config):
    """
    Tests the metric calculation for fine-tuned model outputs.
    It verifies that the integer labels are correctly converted to BIO strings
    and that the seqeval report is generated.
    """
    # --- Setup Mock ---
    # Mock the datamodule to provide a label map
    mock_instance = MagicMock()
    mock_instance.label_map = {"O": 0, "B-FIND": 1, "I-FIND": 2, "B-REG": 3, "I-REG": 4}
    mock_datamodule.return_value = mock_instance

    # --- Act ---
    report = calculate_finetuned_metrics(mock_finetuned_predictions, mock_ner_config)

    # --- Assertions ---
    # Verify that the report has the expected structure from seqeval
    assert "FIND" in report
    assert "REG" in report
    assert "micro avg" in report

    # Check a specific metric to ensure calculation was performed
    # For FIND: TP=2, FP=0, FN=0 -> precision=1.0, recall=1.0, f1=1.0
    # For REG: TP=0, FP=0, FN=1 -> precision=0.0, recall=0.0, f1=0.0
    assert report["FIND"]["precision"] == 1.0
    assert report["REG"]["recall"] == 0.0

def test_calculate_rag_metrics(mock_rag_predictions):
    """
    Tests the metric calculation for RAG model outputs.
    It verifies that the set-based comparison correctly calculates TP, FP, and FN
    to produce the final precision, recall, and F1-score.
    """
    # --- Act ---
    report = calculate_rag_metrics(mock_rag_predictions)

    # --- Assertions ---
    assert "FIND" in report
    assert "REG" in report
    assert "micro avg" in report

    # Based on the mock data:
    # FIND: TP=1, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
    # REG: TP=0, FP=1, FN=1 -> P=0.0, R=0.0, F1=0.0
    assert report["FIND"]["precision"] == 1.0
    assert report["FIND"]["recall"] == 1.0
    assert report["FIND"]["f1-score"] == 1.0
    
    assert report["REG"]["precision"] == 0.0
    assert report["REG"]["recall"] == 0.0
    assert report["REG"]["f1-score"] == 0.0

    # Micro average: Total TP=1, Total FP=1, Total FN=1
    # Micro P = 1 / (1+1) = 0.5
    # Micro R = 1 / (1+1) = 0.5
    # Micro F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
    assert report["micro avg"]["precision"] == 0.5
    assert report["micro avg"]["recall"] == 0.5
    assert report["micro avg"]["f1-score"] == 0.5