# tests/unit/scripts/data/test_inject_noise.py
import pytest
import json
import random
from pathlib import Path

# Add the project root to the Python path
import sys
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from scripts.data.inject_noise import (
    read_jsonl,
    save_jsonl,
    extract_entity_labels,
    inject_noise_to_record
)

@pytest.fixture
def mock_dataset():
    """Returns a mock dataset with entities and relations."""
    return [
        {
            "text": "Nódulo periareolar derecho bien delimitado.",
            "entities": [
                {"id": 1, "label": "HALL", "start_offset": 0, "end_offset": 6},
                {"id": 2, "label": "REG", "start_offset": 7, "end_offset": 26},
                {"id": 3, "label": "CARACT", "start_offset": 27, "end_offset": 45}
            ],
            "relations": [
                {"from_id": 1, "to_id": 2, "type": "ubicar"},
                {"from_id": 1, "to_id": 3, "type": "describir"}
            ]
        },
        {
            "text": "Mamas densas.",
            "entities": [
                {"id": 10, "label": "MAMAS", "start_offset": 0, "end_offset": 5},
                {"id": 11, "label": "DENS", "start_offset": 6, "end_offset": 12}
            ],
            "relations": []
        }
    ]

def test_extract_entity_labels(mock_dataset):
    """Tests that all unique entity labels are correctly extracted."""
    labels = extract_entity_labels(mock_dataset)
    assert len(labels) == 5
    assert set(labels) == {"HALL", "REG", "CARACT", "MAMAS", "DENS"}

def test_read_save_jsonl(tmp_path, mock_dataset):
    """Tests saving and reading JSONL utility functions."""
    file_path = tmp_path / "test_noise_io.jsonl"
    save_jsonl(mock_dataset, file_path)
    
    assert file_path.exists()
    loaded_data = read_jsonl(file_path)
    assert len(loaded_data) == len(mock_dataset)
    assert loaded_data[0]["text"] == mock_dataset[0]["text"]
    assert len(loaded_data[0]["entities"]) == 3

def test_inject_no_noise(mock_dataset):
    """Tests that setting probabilities to 0.0 results in no modifications."""
    random.seed(42)
    record = mock_dataset[0]
    entity_labels = extract_entity_labels(mock_dataset)
    
    perturbed = inject_noise_to_record(
        record,
        entity_labels=entity_labels,
        label_swap_prob=0.0,
        offset_shift_prob=0.0,
        relation_drop_prob=0.0
    )
    
    assert perturbed["text"] == record["text"]
    assert perturbed["entities"] == record["entities"]
    assert perturbed["relations"] == record["relations"]

def test_inject_relation_omission(mock_dataset):
    """Tests dropping relations with probability 1.0."""
    random.seed(42)
    record = mock_dataset[0]
    entity_labels = extract_entity_labels(mock_dataset)
    
    perturbed = inject_noise_to_record(
        record,
        entity_labels=entity_labels,
        label_swap_prob=0.0,
        offset_shift_prob=0.0,
        relation_drop_prob=1.0
    )
    
    assert len(perturbed["relations"]) == 0
    assert len(record["relations"]) == 2  # Original should be untouched

def test_inject_label_swap(mock_dataset):
    """Tests swapping labels with probability 1.0."""
    random.seed(42)
    record = mock_dataset[0]
    entity_labels = ["HALL", "OTHER"]  # Only these two labels are swap candidates
    
    perturbed = inject_noise_to_record(
        record,
        entity_labels=entity_labels,
        label_swap_prob=1.0,
        offset_shift_prob=0.0,
        relation_drop_prob=0.0
    )
    
    # Entity 1 had label "HALL", should be swapped to "OTHER" since it is the only alternative
    assert perturbed["entities"][0]["label"] == "OTHER"
    # Entity 2 had label "REG". Since "REG" is not in entity_labels, the available alternatives to swap to
    # are ["HALL", "OTHER"]. With seed 42, it picked "HALL" (or "OTHER" depending on choice).
    assert perturbed["entities"][1]["label"] in ["HALL", "OTHER"]

def test_inject_offset_shift_boundaries(mock_dataset):
    """Tests boundary safety of entity offset boundary shifts."""
    random.seed(42)
    # A short record where shifting could exceed boundaries easily
    record = {
        "text": "ABC",
        "entities": [
            {"label": "FIND", "start_offset": 0, "end_offset": 1}  # "A"
        ]
    }
    entity_labels = ["FIND"]
    
    # Run multiple times with seed to verify offsets stay within bounds [0, 3]
    for seed in range(50):
        random.seed(seed)
        perturbed = inject_noise_to_record(
            record,
            entity_labels=entity_labels,
            label_swap_prob=0.0,
            offset_shift_prob=1.0,
            relation_drop_prob=0.0
        )
        
        ent = perturbed["entities"][0]
        start = ent["start_offset"]
        end = ent["end_offset"]
        
        # Verify valid bounds
        assert start >= 0, f"start_offset {start} is negative"
        assert end <= 3, f"end_offset {end} exceeds text length 3"
        assert start < end, f"start_offset {start} >= end_offset {end}"
