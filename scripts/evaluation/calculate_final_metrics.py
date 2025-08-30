import argparse
import json
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import classification_report as sklearn_classification_report
import numpy as np

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
    """Loads prediction records from a .jsonl file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def calculate_ner_metrics(predictions: list) -> dict:
    """
    Calculates NER metrics for any model by comparing sets of entity dictionaries.
    This function serves as the unified metric calculator for both fine-tuned and RAG models.

    Args:
        predictions (list): The list of prediction records. Each record must contain
                            'true_entities' and 'predicted_entities' keys.

    Returns:
        dict: A classification report dictionary.
    """
    entity_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for record in predictions:
        # Convert true and predicted entities to sets of tuples for easy comparison
        true_entities_set = {(e['text'].strip(), e['label']) for e in record['true_entities']}
        predicted_entities_set = {(e['text'].strip(), e['label']) for e in record['predicted_entities']}

        # Calculate True Positives, False Positives, and False Negatives for this record
        tp = true_entities_set.intersection(predicted_entities_set)
        fp = predicted_entities_set - true_entities_set
        fn = true_entities_set - predicted_entities_set

        # Aggregate statistics per entity type
        for _, label in tp:
            entity_metrics[label]['tp'] += 1
        for _, label in fp:
            entity_metrics[label]['fp'] += 1
        for _, label in fn:
            entity_metrics[label]['fn'] += 1

    # --- Calculate the final report from the aggregated statistics ---
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
    
    # Calculate weighted average (average weighted by support)
    weighted_f1 = sum(report[label]['f1-score'] * report[label]['support'] for label in sorted_labels) / total_support if total_support > 0 else 0
    report['weighted avg'] = {
        'precision': sum(report[label]['precision'] * report[label]['support'] for label in sorted_labels) / total_support if total_support > 0 else 0,
        'recall': sum(report[label]['recall'] * report[label]['support'] for label in sorted_labels) / total_support if total_support > 0 else 0,
        'f1-score': weighted_f1,
        'support': total_support
    }

    return report

def calculate_re_metrics(predictions: list) -> dict:
    """
    Calculates RE metrics by comparing sets of relation dictionaries.
    This function serves as the unified metric calculator for both fine-tuned and RAG models.

    Args:
        predictions (list): The list of prediction records. Each record must contain
                            'true_relations' and 'predicted_relations' keys.

    Returns:
        dict: A classification report dictionary.
    """
    relation_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for record in predictions:
        # Convert relations to a set of tuples for comparison: (from_id, to_id, type)
        true_relations_set = {
            (r['from_id'], r['to_id'], r['type']) for r in record['true_relations']
        }
        predicted_relations_set = {
            (r['from_id'], r['to_id'], r['type']) for r in record['predicted_relations']
        }

        tp = true_relations_set.intersection(predicted_relations_set)
        fp = predicted_relations_set - true_relations_set
        fn = true_relations_set - predicted_relations_set

        # Aggregate statistics per relation type
        for _, _, rel_type in tp:
            relation_metrics[rel_type]['tp'] += 1
        for _, _, rel_type in fp:
            relation_metrics[rel_type]['fp'] += 1
        for _, _, rel_type in fn:
            relation_metrics[rel_type]['fn'] += 1

    # --- Calculate the final report from the aggregated statistics ---
    report = {}
    all_tp, all_fp, all_fn = 0, 0, 0
    total_support = 0

    sorted_labels = sorted(relation_metrics.keys())
    for label in sorted_labels:
        tp = relation_metrics[label]['tp']
        fp = relation_metrics[label]['fp']
        fn = relation_metrics[label]['fn']
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

    # Calculate micro average
    micro_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    micro_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    report['micro avg'] = {
        'precision': micro_precision,
        'recall': micro_recall,
        'f1-score': micro_f1,
        'support': total_support
    }

    # Calculate weighted average
    report['weighted avg'] = {
        'precision': sum(report[label]['precision'] * report[label]['support'] for label in sorted_labels) / total_support if total_support > 0 else 0,
        'recall': sum(report[label]['recall'] * report[label]['support'] for label in sorted_labels) / total_support if total_support > 0 else 0,
        'f1-score': sum(report[label]['f1-score'] * report[label]['support'] for label in sorted_labels) / total_support if total_support > 0 else 0,
        'support': total_support
    }

    return report

def aggregate_metrics(reports: list) -> dict:
    """
    Aggregates metrics from multiple reports to calculate mean and standard deviation.

    Args:
        reports (list): A list of classification report dictionaries.

    Returns:
        dict: A dictionary containing the mean and std dev for key metrics.
    """
    if not reports:
        return {}

    # We focus on the 'weighted avg' as the primary summary statistic
    key_metrics = ['precision', 'recall', 'f1-score']

    # Extract the weighted average scores from each report
    weighted_avg_scores = {metric: [] for metric in key_metrics}
    for report in reports:
        if 'weighted avg' in report:
            for metric in key_metrics:
                weighted_avg_scores[metric].append(report['weighted avg'][metric])

    # Calculate mean and standard deviation for each metric
    summary = {}
    for metric in key_metrics:
        scores = weighted_avg_scores[metric]
        if scores:
            summary[metric] = {
                "mean": np.mean(scores),
                "std": np.std(scores)
            }

    return summary

def main(prediction_path: str, prediction_dir: str, eval_type: str, output_path: str):
    """
    Main function to calculate and save metrics from a prediction file or directory.
    """
    if prediction_dir:
        print(f"--- Calculating Aggregate Metrics for Directory: {prediction_dir} ---")
        prediction_files = list(Path(prediction_dir).glob('*.jsonl'))
        if not prediction_files:
            raise FileNotFoundError(f"No .jsonl prediction files found in '{prediction_dir}'.")

        print(f"Found {len(prediction_files)} prediction files to process.")

        individual_reports = []
        for file_path in prediction_files:
            report = process_single_file(str(file_path), eval_type)
            individual_reports.append({
                "source_file": file_path.name,
                "report": report
            })

        aggregate_summary = aggregate_metrics([r['report'] for r in individual_reports])

        final_report = {
            "aggregate_summary": aggregate_summary,
            "individual_reports": individual_reports
        }

    else:
        print(f"--- Calculating Metrics for File: {prediction_path} ---")
        final_report = process_single_file(prediction_path, eval_type)

    # Save the final report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(final_report, f, indent=4)

    print("\n--- Final Metrics Report ---")
    print(json.dumps(final_report, indent=4))
    print(f"\nReport saved successfully to: {output_path}")


def process_single_file(prediction_path: str, eval_type: str) -> dict:
    """Processes a single prediction file and returns its metrics report."""
    predictions = load_predictions(prediction_path)

    if eval_type in ['ner', 'rag']:
        report = calculate_ner_metrics(predictions)
    elif eval_type == 're':
        report = calculate_re_metrics(predictions)
    else:
        raise ValueError(f"Unknown evaluation type: '{eval_type}'. Must be 'ner', 'rag', or 're'.")

    return convert_numpy_types(report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate final metrics from a single prediction file or a directory of files."
    )

    # Create a mutually exclusive group for file or directory input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--prediction-path', 
        type=str,
        help='Path to a single .jsonl file containing predictions.'
    )
    group.add_argument(
        '--prediction-dir',
        type=str,
        help='Path to a directory containing multiple .jsonl prediction files.'
    )

    parser.add_argument(
        '--type',
        type=str,
        required=True,
        choices=['ner', 'rag', 're'],
        help="The type of task evaluation."
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help="Path to save the final JSON metrics report."
    )

    args = parser.parse_args()

    main(args.prediction_path, args.prediction_dir, args.type, args.output_path)