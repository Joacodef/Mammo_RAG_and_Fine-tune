# scripts/run_evaluation.py
import argparse
import yaml
from pathlib import Path
import json
import torch
import numpy as np
from collections import defaultdict
from seqeval.metrics import classification_report as ner_classification_report
from sklearn.metrics import classification_report as re_classification_report
from datetime import datetime
import shutil

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_loader.ner_datamodule import NERDataModule
from src.data_loader.re_datamodule import REDataModule
from src.models.ner_bert import BertNerModel
from src.models.re_model import REModel
from src.evaluation.predictor import Predictor

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

def run_evaluation(config):
    """
    Main function to run evaluation on a test set for either NER or RE.
    """

    task = config.get('task')
    if task not in ['ner', 're']:
        raise ValueError("Configuration file must specify a 'task': 'ner' or 're'.")

    model_path = config['model_path']
    test_file = config['test_file']
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. Initialize Task-Specific Modules ---
    if task == 'ner':
        print("Initializing NER evaluation components...")
        # For evaluation, the datamodule needs the config to find the entity labels
        datamodule = NERDataModule(config=config, test_file=test_file)
        datamodule.setup()
        # For evaluation, load the model directly without specifying n_labels.
        # The number of labels is already stored in the model's config.json.
        model = BertNerModel(base_model=model_path)
        inv_label_map = {v: k for k, v in datamodule.label_map.items()}
    else: # task == 're'
        print("Initializing RE evaluation components...")
        datamodule = REDataModule(config=config, test_file=test_file)
        datamodule.setup()
        # For RE, the tokenizer is still needed to handle special tokens, but n_labels is not.
        model = REModel(base_model=model_path, tokenizer=datamodule.tokenizer)
        inv_label_map = {v: k for k, v in datamodule.relation_map.items()}

    test_loader = datamodule.test_dataloader()

    # --- 3. Initialize Predictor ---
    print(f"Loading model from: {model_path}")
    predictor = Predictor(model=model, device=device)

    # --- 4. Get Predictions ---
    predictions, true_labels = predictor.predict(test_loader, task_type=task)

    # --- 5. Calculate and Save Metrics ---
    print("\n--- Evaluation Results ---")
    if task == 'ner':
        true_labels_str = [[inv_label_map.get(l, "O") for l in seq] for seq in true_labels]
        pred_labels_str = [[inv_label_map.get(p, "O") for p in pred_seq] for pred_seq in predictions]
        report = ner_classification_report(true_labels_str, pred_labels_str, output_dict=True, zero_division=0)
    else: # task == 're'
        true_labels_str = [inv_label_map.get(l, "No_Relation") for l in true_labels]
        pred_labels_str = [inv_label_map.get(p, "No_Relation") for p in predictions]
        report = re_classification_report(true_labels_str, pred_labels_str, output_dict=True, zero_division=0)

    # Convert numpy types to native Python types for JSON serialization
    report = convert_numpy_types(report)

    # Save and print the report
    report_path = output_dir / f"evaluation_metrics_{Path(model_path).name}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(json.dumps(report, indent=4))
    print(f"\nEvaluation metrics saved to: {report_path}")
    print(f"\nEvaluation for '{Path(model_path).name}' finished successfully.")
    return report


def aggregate_and_save_metrics(all_reports, output_dir):
    """
    Aggregates metrics from multiple reports to calculate mean and standard deviation.

    Args:
        all_reports (list): A list of classification report dictionaries.
        output_dir (Path): The directory to save the final summary file.
    """
    # Use defaultdict to easily append scores for each metric
    aggregated_metrics = defaultdict(lambda: defaultdict(list))
    
    # --- 1. Collect all scores from all reports ---
    for report in all_reports:
        for label, metrics in report.items():
            # Skip non-dict items like 'accuracy' which is a flat value
            if isinstance(metrics, dict):
                aggregated_metrics[label]['precision'].append(metrics.get('precision', 0))
                aggregated_metrics[label]['recall'].append(metrics.get('recall', 0))
                aggregated_metrics[label]['f1-score'].append(metrics.get('f1-score', 0))

    # --- 2. Calculate statistics (mean and std) ---
    summary_report = {}
    for label, metrics in aggregated_metrics.items():
        summary_report[label] = {
            'precision': {
                'mean': np.mean(metrics['precision']),
                'std': np.std(metrics['precision'])
            },
            'recall': {
                'mean': np.mean(metrics['recall']),
                'std': np.std(metrics['recall'])
            },
            'f1-score': {
                'mean': np.mean(metrics['f1-score']),
                'std': np.std(metrics['f1-score'])
            }
        }
    
    # --- 3. Save the final aggregated report ---
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=4)
        
    print("\n--- Aggregated Evaluation Summary ---")
    print(json.dumps(summary_report, indent=4))
    print(f"\nAggregated summary saved to: {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a batch of evaluations for all models in a given directory."
    )
    
    parser.add_argument(
        '--config-path', 
        type=str, 
        required=True, 
        help='Path to the YAML configuration file for evaluation.'
    )
    
    args = parser.parse_args()
    
    # --- 1. Load Base Configuration ---
    with open(args.config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    model_dir = Path(base_config['model_dir'])
    sample_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('sample-')])

    if not sample_dirs:
        raise FileNotFoundError(f"No 'sample-*' directories found in '{model_dir}'.")

    # --- 2. Create Dynamic, Timestamped Output Directory ---
    task = base_config.get('task', 'unknown_task')
    train_size_folder_name = model_dir.parent.name  # e.g., "train-50"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define the new unique output directory path
    output_dir_timestamped = Path(base_config['output_dir']) / train_size_folder_name / timestamp
    output_dir_timestamped.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved in: {output_dir_timestamped}")

    # Save a copy of the evaluation configuration for reproducibility
    shutil.copy(args.config_path, output_dir_timestamped / "evaluation_config.yaml")

    all_individual_reports = []
    print(f"Found {len(sample_dirs)} model samples to evaluate in '{model_dir}'.")

    # --- 3. Loop and Evaluate Each Model Sample ---
    for i, sample_path in enumerate(sample_dirs):
        print(f"\n{'='*20} Evaluating: {sample_path.name} ({i+1}/{len(sample_dirs)}) {'='*20}")
        
        # Create a dynamic config for the current sample, injecting the new output path
        sample_config = base_config.copy()
        sample_config['model_path'] = str(sample_path)
        sample_config['output_dir'] = str(output_dir_timestamped) # Override with timestamped path
        
        report = run_evaluation(sample_config)
        all_individual_reports.append(report)

    # --- 4. Aggregate and Save Final Summary ---
    if all_individual_reports:
        aggregate_and_save_metrics(all_individual_reports, output_dir_timestamped)

    print("\nBatch evaluation finished successfully.")