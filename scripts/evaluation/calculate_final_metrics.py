import argparse
import json
from pathlib import Path
import yaml
from collections import defaultdict
from seqeval.metrics import classification_report as ner_classification_report
from sklearn.metrics import classification_report as sklearn_classification_report
import numpy as np

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_loader.ner_datamodule import NERDataModule
from src.data_loader.re_datamodule import REDataModule

def convert_numpy_types(obj):
    """
    Recursively converts numpy number types in a dictionary to native Python types
    to ensure JSON serialization compatibility.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.intc, np.intp, np.int8,
                    np.int16, np.int32, np.int64, np.uint8,
                    np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def load_predictions(file_path: str) -> list:
    """Loads raw prediction records from a .jsonl file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def calculate_finetuned_metrics(predictions: list, config: dict, test_file: str) -> dict:
    """
    Calculates NER metrics for fine-tuned models using seqeval.

    Args:
        predictions (list): The list of raw prediction records.
        config (dict): The evaluation configuration, needed for the label map.
        test_file (str): The path to the test data file.

    Returns:
        dict: A classification report dictionary from seqeval.
    """
    # We need a DataModule instance to get the inverse label map
    datamodule = NERDataModule(config=config, test_file=test_file)
    datamodule.setup()
    inv_label_map = {v: k for k, v in datamodule.label_map.items()}

    true_labels_str = []
    pred_labels_str = []

    for record in predictions:
        true_labels_str.append([inv_label_map.get(l, "O") for l in record['true_labels']])
        pred_labels_str.append([inv_label_map.get(p, "O") for p in record['predicted_labels']])

    return ner_classification_report(
        true_labels_str,
        pred_labels_str,
        output_dict=True,
        zero_division=0
    )

def calculate_rag_metrics(predictions: list) -> dict:
    """
    Calculates NER metrics for RAG models by comparing sets of entities.

    Args:
        predictions (list): The list of raw prediction records from the RAG pipeline.

    Returns:
        dict: A classification report dictionary in the same format as seqeval.
    """
    entity_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for record in predictions:
        # For true entities, extract the text from the source using offsets
        true_entities_set = set()
        for entity in record['true_entities']:
            text = record['source_text'][entity['start_offset']:entity['end_offset']]
            true_entities_set.add((text.strip(), entity['label']))

        # Predicted entities are already in the correct format
        predicted_entities_set = set()
        for entity in record['predicted_entities']:
            predicted_entities_set.add((entity['text'].strip(), entity['label']))

        # Calculate TP, FP, FN for this record
        tp = true_entities_set.intersection(predicted_entities_set)
        fp = predicted_entities_set - true_entities_set
        fn = true_entities_set - predicted_entities_set

        # Aggregate stats per entity type
        for entity in tp:
            entity_metrics[entity[1]]['tp'] += 1
        for entity in fp:
            entity_metrics[entity[1]]['fp'] += 1
        for entity in fn:
            entity_metrics[entity[1]]['fn'] += 1

# Calculate final report
    report = {}
    all_tp, all_fp, all_fn = 0, 0, 0
    total_support = 0

    sorted_labels = sorted(entity_metrics.keys())
    for label in sorted_labels:
        tp = entity_metrics[label]['tp']
        fp = entity_metrics[label]['fp']
        fn = entity_metrics[label]['fn']
        support = tp + fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        report[label] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        }
        all_tp += tp
        all_fp += fp
        all_fn += fn
        total_support += support
    
    # Calculate micro average (overall performance)
    micro_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    micro_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    report['micro avg'] = {
        'precision': micro_precision,
        'recall': micro_recall,
        'f1-score': micro_f1,
        'support': total_support
    }

    # Calculate macro average (unweighted average of per-class metrics)
    macro_precision = sum(report[label]['precision'] for label in sorted_labels) / len(sorted_labels) if sorted_labels else 0
    macro_recall = sum(report[label]['recall'] for label in sorted_labels) / len(sorted_labels) if sorted_labels else 0
    macro_f1 = sum(report[label]['f1-score'] for label in sorted_labels) / len(sorted_labels) if sorted_labels else 0

    report['macro avg'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1-score': macro_f1,
        'support': total_support
    }

    # Calculate weighted average (average weighted by support)
    weighted_precision = sum(report[label]['precision'] * report[label]['support'] for label in sorted_labels) / total_support if total_support > 0 else 0
    weighted_recall = sum(report[label]['recall'] * report[label]['support'] for label in sorted_labels) / total_support if total_support > 0 else 0
    weighted_f1 = sum(report[label]['f1-score'] * report[label]['support'] for label in sorted_labels) / total_support if total_support > 0 else 0

    report['weighted avg'] = {
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1-score': weighted_f1,
        'support': total_support
    }

    return report

def calculate_finetuned_re_metrics(predictions: list, config: dict) -> dict:
    """
    Calculates RE metrics for fine-tuned models using scikit-learn.

    Args:
        predictions (list): The list of raw prediction records.
        config (dict): The evaluation configuration, needed for the label map.

    Returns:
        dict: A classification report dictionary from scikit-learn.
    """
    true_labels = [record['true_labels'] for record in predictions]
    pred_labels = [record['predicted_labels'] for record in predictions]

    relation_labels = config.get('model', {}).get('relation_labels', [])
    
    # Create the integer list of all possible labels. This is the key change.
    label_ids = list(range(len(relation_labels)))

    return sklearn_classification_report(
        true_labels,
        pred_labels,
        labels=label_ids, # Explicitly provide all possible label IDs
        target_names=relation_labels,
        output_dict=True,
        zero_division=0
    )



def main(prediction_path: str, eval_type: str, config_path: str, output_path: str, test_file: str):
    """
    Main function to calculate and save metrics from a raw prediction file.
    """
    print(f"--- Calculating Metrics for: {prediction_path} ---")
    print(f"Evaluation type: {eval_type}")

    predictions = load_predictions(prediction_path)

    if eval_type == 'finetuned_ner':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        report = calculate_finetuned_metrics(predictions, config, test_file)

    elif eval_type == 'finetuned_re':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        report = calculate_finetuned_re_metrics(predictions, config)

    elif eval_type == 'rag':
        report = calculate_rag_metrics(predictions)

    else:
        raise ValueError(f"Unknown evaluation type: '{eval_type}'. Must be 'finetuned_ner', 'finetuned_re', or 'rag'.")
    
    # Convert all numpy types in the report to native Python types
    report = convert_numpy_types(report)
    
    # Save the final report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)

    print("\n--- Final Metrics Report ---")
    print(json.dumps(report, indent=4))
    print(f"\nReport saved successfully to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate final metrics from raw prediction files."
    )
    
    parser.add_argument(
        '--prediction-path', 
        type=str, 
        required=True,
        help='Path to the .jsonl file containing raw predictions.'
    )
    parser.add_argument(
        '--type',
        type=str,
        required=True,
        choices=['finetuned_ner', 'finetuned_re', 'rag'],
        help="The type of model that generated the predictions."
    )
    parser.add_argument(
        '--config-path',
        type=str,
        default='configs/evaluation_ner_config.yaml',
        help="Path to the evaluation config file (required for 'finetuned' type)."
    )
    parser.add_argument(
        '--test-file',
        type=str,
        default='data/processed/test.jsonl',
        help="Path to the test data file (required for 'finetuned' type to get the label map)."
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help="Path to save the final JSON metrics report."
    )
    
    args = parser.parse_args()
    main(args.prediction_path, args.type, args.config_path, args.output_path, args.test_file)