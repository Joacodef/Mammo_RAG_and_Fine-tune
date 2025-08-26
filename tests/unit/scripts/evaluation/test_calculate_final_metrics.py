# tests/unit/scripts/evaluation/test_calculate_final_metrics.py
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.evaluation.calculate_final_metrics import calculate_ner_metrics

# --- Fixtures for Testing ---

@pytest.fixture
def mock_unified_predictions():
    """
    Provides mock prediction data in the new, unified format.
    This format is used by both RAG and fine-tuned pipelines.
    """
    return [
        {
            "source_text": "Nódulo en mama derecha.",
            "true_entities": [
                {"text": "Nódulo", "label": "FIND"},
                {"text": "mama derecha", "label": "REG"}
            ],
            "predicted_entities": [
                {"text": "Nódulo", "label": "FIND"},       # True Positive (FIND)
                {"text": "mama izquierda", "label": "REG"}  # False Positive (REG)
            ]
            # This setup also implies one FN for "mama derecha" (REG)
        },
        {
            "source_text": "Sin hallazgos de jerarquía.",
            "true_entities": [], # No true entities in this record
            "predicted_entities": [
                {"text": "hallazgos de jerarquía", "label": "FIND"} # False Positive (FIND)
            ]
        }
    ]

# --- Test Cases ---

def test_calculate_ner_metrics(mock_unified_predictions):
    """
    Tests the unified metric calculation for NER outputs.
    It verifies that the set-based comparison correctly calculates TP, FP, and FN
    to produce the final precision, recall, and F1-score.
    """
    # --- Act ---
    report = calculate_ner_metrics(mock_unified_predictions)

    # --- Assertions ---
    assert "FIND" in report
    assert "REG" in report
    assert "micro avg" in report
    assert "weighted avg" in report

    # Based on the mock data:
    # FIND: TP=1, FP=1, FN=0 -> P=0.5, R=1.0, F1=2/3, Support=1
    assert report["FIND"]["precision"] == 0.5
    assert report["FIND"]["recall"] == 1.0
    assert report["FIND"]["f1-score"] == pytest.approx(2/3)
    assert report["FIND"]["support"] == 1
    
    # REG: TP=0, FP=1, FN=1 -> P=0.0, R=0.0, F1=0.0, Support=1
    assert report["REG"]["precision"] == 0.0
    assert report["REG"]["recall"] == 0.0
    assert report["REG"]["f1-score"] == 0.0
    assert report["REG"]["support"] == 1

    # Micro average: Total TP=1, Total FP=2, Total FN=1 -> P=1/3, R=0.5, F1=0.4, Support=2
    assert report["micro avg"]["precision"] == pytest.approx(1/3)
    assert report["micro avg"]["recall"] == 0.5
    assert report["micro avg"]["f1-score"] == 0.4
    assert report["micro avg"]["support"] == 2
    
    # Weighted average F1 = (F1_FIND * Sup_FIND + F1_REG * Sup_REG) / Total_Sup
    #                    = ((2/3 * 1) + (0.0 * 1)) / 2 = 1/3
    assert report["weighted avg"]["f1-score"] == pytest.approx(1/3)