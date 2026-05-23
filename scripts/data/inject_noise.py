# scripts/data/inject_noise.py
import json
import random
import argparse
from pathlib import Path
import sys

def read_jsonl(file_path):
    """Reads a .jsonl file and returns a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """Saves a list of dictionaries to a .jsonl file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def extract_entity_labels(data):
    """Extracts all unique entity labels present in the dataset."""
    labels = set()
    for record in data:
        for entity in record.get("entities", []):
            if "label" in entity:
                labels.add(entity["label"])
    return sorted(list(labels))

def inject_noise_to_record(record, entity_labels, label_swap_prob, offset_shift_prob, relation_drop_prob):
    """
    Applies noise to a single mammography report record:
    - Entity label swapping
    - Entity offset boundary perturbation
    - Relation omission (deletion)
    """
    perturbed_record = {
        "text": record["text"]
    }
    
    text_len = len(record["text"])
    
    # 1. Perturb Entities
    perturbed_entities = []
    for entity in record.get("entities", []):
        new_entity = entity.copy()
        
        # Label swapping
        if label_swap_prob > 0.0 and len(entity_labels) > 1:
            if random.random() < label_swap_prob:
                current_label = entity.get("label")
                alternatives = [lbl for lbl in entity_labels if lbl != current_label]
                if alternatives:
                    new_entity["label"] = random.choice(alternatives)
                    
        # Offset boundary perturbation
        if offset_shift_prob > 0.0:
            if random.random() < offset_shift_prob:
                start = entity.get("start_offset")
                end = entity.get("end_offset")
                if isinstance(start, int) and isinstance(end, int):
                    # We pick shifts in [-3, 3] excluding 0
                    shift_options = [-3, -2, -1, 1, 2, 3]
                    shift_start = random.choice(shift_options)
                    shift_end = random.choice(shift_options)
                    
                    new_start = max(0, start + shift_start)
                    new_end = min(text_len, end + shift_end)
                    
                    # Ensure start < end, otherwise try to keep original length or fallback
                    if new_start < new_end:
                        new_entity["start_offset"] = new_start
                        new_entity["end_offset"] = new_end
                    # If invalid, keep original offsets to avoid corruption
        
        perturbed_entities.append(new_entity)
    
    perturbed_record["entities"] = perturbed_entities
    
    # 2. Perturb Relations (Omission)
    perturbed_relations = []
    for relation in record.get("relations", []):
        if relation_drop_prob > 0.0:
            if random.random() < relation_drop_prob:
                # Omit this relation
                continue
        perturbed_relations.append(relation.copy())
        
    if "relations" in record or perturbed_relations:
        perturbed_record["relations"] = perturbed_relations
        
    return perturbed_record

def main():
    parser = argparse.ArgumentParser(description="Inject annotation noise into a mammography dataset to evaluate RAG robustness.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the clean source .jsonl dataset.")
    parser.add_argument("--output-path", type=str, required=True, help="Path where the noisy .jsonl dataset will be saved.")
    parser.add_argument("--label-swap-prob", type=float, default=0.0, help="Probability of swapping an entity's label with an alternative.")
    parser.add_argument("--offset-shift-prob", type=float, default=0.0, help="Probability of perturbing an entity's character offset boundaries by [-3, 3] chars.")
    parser.add_argument("--relation-drop-prob", type=float, default=0.0, help="Probability of omitting/dropping a relation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)
        
    print(f"Reading clean dataset from: {input_path}")
    data = read_jsonl(input_path)
    print(f"Loaded {len(data)} records.")
    
    # Determine the vocabulary of entity labels across the dataset for swapping
    entity_labels = extract_entity_labels(data)
    print(f"Inferred entity label vocabulary ({len(entity_labels)} labels): {entity_labels}")
    
    noisy_data = []
    swapped_count = 0
    shifted_count = 0
    relations_before = 0
    relations_after = 0
    
    for record in data:
        # Counters for summary reporting
        for ent in record.get("entities", []):
            pass # just checking
            
        relations_before += len(record.get("relations", []))
        
        noisy_record = inject_noise_to_record(
            record,
            entity_labels=entity_labels,
            label_swap_prob=args.label_swap_prob,
            offset_shift_prob=args.offset_shift_prob,
            relation_drop_prob=args.relation_drop_prob
        )
        
        # Log counts of changes
        for orig_ent, new_ent in zip(record.get("entities", []), noisy_record.get("entities", [])):
            if orig_ent.get("label") != new_ent.get("label"):
                swapped_count += 1
            if orig_ent.get("start_offset") != new_ent.get("start_offset") or orig_ent.get("end_offset") != new_ent.get("end_offset"):
                shifted_count += 1
                
        relations_after += len(noisy_record.get("relations", []))
        noisy_data.append(noisy_record)
        
    # Print metrics
    total_entities = sum(len(r.get("entities", [])) for r in data)
    print("\n--- Noise Injection Summary ---")
    if total_entities > 0:
        print(f"Label Swaps: {swapped_count} / {total_entities} entities ({swapped_count/total_entities:.2%})")
        print(f"Offset Shifts: {shifted_count} / {total_entities} entities ({shifted_count/total_entities:.2%})")
    else:
        print("No entities found in the dataset.")
        
    if relations_before > 0:
        dropped_relations = relations_before - relations_after
        print(f"Relation Omissions: {dropped_relations} / {relations_before} relations ({dropped_relations/relations_before:.2%})")
    else:
        print("No relations found in the dataset.")
        
    print(f"Saving noisy dataset to: {args.output_path}")
    save_jsonl(noisy_data, args.output_path)
    print("Done!")

if __name__ == '__main__':
    main()
