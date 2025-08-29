# tests/unit/scripts/evaluation/test_postprocess_rag_predictions.py
import pytest
import json
from unittest.mock import patch, mock_open

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.evaluation.postprocess_rag_predictions import find_nearest_match, postprocess_predictions

# --- Fixtures for Testing ---

@pytest.fixture
def sample_record_needs_correction():
    """Provides a sample record where the entity offset is incorrect."""
    return {
        "source_text": "Ambas mamas son densas. Se observa un nódulo en la mama derecha.",
        "predicted_entities": [{
            "label": "FIND",
            "start_offset": 30, # Incorrect, points into "observa un"
            "end_offset": 38,
            "text": "nódulo"
        }]
    }

@pytest.fixture
def sample_record_perfect_match():
    """Provides a sample record where the entity offset is already correct."""
    return {
        "source_text": "Ambas mamas son densas. Se observa un nódulo en la mama derecha.",
        "predicted_entities": [{
            "label": "FIND",
            "start_offset": 39, # Correct
            "end_offset": 45,
            "text": "nódulo"
        }]
    }

@pytest.fixture
def sample_record_no_match():
    """Provides a sample record where the entity text is not in the source text."""
    return {
        "source_text": "Ambas mamas son densas.",
        "predicted_entities": [{
            "label": "FIND",
            "start_offset": 10,
            "end_offset": 18,
            "text": "hallazgo" # This word is not in the source_text
        }]
    }

@pytest.fixture
def sample_record_multiple_occurrences():
    """Provides a sample record with multiple occurrences of the same entity text."""
    return {
        "source_text": "Nódulo en mama izquierda. Se revisa el nódulo contralateral.",
        "predicted_entities": [{
            "label": "FIND",
            "start_offset": 5, # Hint points to the first one
            "end_offset": 11,
            "text": "Nódulo"
        }, {
            "label": "FIND",
            "start_offset": 40, # Hint points to the second one
            "end_offset": 46,
            "text": "nódulo"
        }]
    }

# --- Tests for find_nearest_match helper function ---

def test_find_nearest_match_perfect_match():
    """Tests that the correct offsets are returned when the hint is accurate."""
    text = "Se observa un nódulo."
    substring = "nódulo"
    start_hint = 14
    start, end = find_nearest_match(substring, text, start_hint)
    assert start == 14
    assert end == 20

def test_find_nearest_match_correction_needed():
    """Tests that the function corrects a slightly inaccurate offset hint."""
    text = "Se observa un nódulo."
    substring = "nódulo"
    start_hint = 10 # Hint is a bit early
    start, end = find_nearest_match(substring, text, start_hint)
    assert start == 14
    assert end == 20

def test_find_nearest_match_no_match_found():
    """Tests that None is returned when the substring does not exist in the text."""
    text = "Se observa una asimetría."
    substring = "nódulo"
    start_hint = 14
    start, end = find_nearest_match(substring, text, start_hint)
    assert start is None
    assert end is None

def test_find_nearest_match_selects_closest_occurrence():
    """Tests that the function selects the occurrence nearest to the hint."""
    text = "Un nódulo aquí, y otro nódulo por allá."
    substring = "nódulo"
    
    # Hint is closer to the second occurrence
    start_hint = 20
    start, end = find_nearest_match(substring, text, start_hint)
    assert start == 23 # "nódulo por allá"
    assert end == 29

    # Hint is closer to the first occurrence
    start_hint = 5
    start, end = find_nearest_match(substring, text, start_hint)
    assert start == 3 # "nódulo aquí"
    assert end == 9

# --- Tests for postprocess_predictions main function ---

@patch("builtins.open", new_callable=mock_open)
def test_postprocess_predictions_corrects_offsets(mock_file, sample_record_needs_correction):
    """
    Tests that the main function reads a file, corrects the offsets, and writes the result.
    """
    # Simulate reading the input file
    mock_file.return_value.__iter__.return_value = [json.dumps(sample_record_needs_correction)]

    postprocess_predictions("dummy_input.jsonl", "dummy_output.jsonl")

    # Verify the content written to the output file
    handle = mock_file()
    written_data = handle.write.call_args[0][0]
    corrected_record = json.loads(written_data)

    corrected_entity = corrected_record["predicted_entities"][0]
    assert corrected_entity["start_offset"] == 39
    assert corrected_entity["end_offset"] == 45
    assert corrected_entity["text"] == "nódulo"

@patch("builtins.open", new_callable=mock_open)
def test_postprocess_predictions_drops_unmatched_entities(mock_file, sample_record_no_match):
    """
    Tests that entities with no match in the source text are dropped from the output.
    """
    mock_file.return_value.__iter__.return_value = [json.dumps(sample_record_no_match)]

    postprocess_predictions("dummy_input.jsonl", "dummy_output.jsonl")

    handle = mock_file()
    written_data = handle.write.call_args[0][0]
    corrected_record = json.loads(written_data)
    
    # The list of predicted entities should now be empty
    assert len(corrected_record["predicted_entities"]) == 0

@patch("builtins.open", new_callable=mock_open)
def test_postprocess_predictions_handles_malformed_entity(mock_file):
    """
    Tests that the function gracefully skips entities with missing or invalid keys.
    """
    malformed_record = {
        "source_text": "Some text.",
        "predicted_entities": [
            {"label": "FIND"}, # Missing text and start_offset
            {"text": "Some text", "start_offset": "invalid"} # Invalid offset type
        ]
    }
    mock_file.return_value.__iter__.return_value = [json.dumps(malformed_record)]

    # The function should run without raising an error
    postprocess_predictions("dummy_input.jsonl", "dummy_output.jsonl")
    
    handle = mock_file()
    written_data = handle.write.call_args[0][0]
    corrected_record = json.loads(written_data)
    
    # Both malformed entities should have been dropped
    assert len(corrected_record["predicted_entities"]) == 0