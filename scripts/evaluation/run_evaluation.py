# scripts/run_evaluation.py
import argparse
import yaml
from pathlib import Path
import json
import torch
import numpy as np
from seqeval.metrics import classification_report as ner_classification_report
from sklearn.metrics import classification_report as re_classification_report

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

def run_evaluation(config_path):
    """
    Main function to run evaluation on a test set for either NER or RE.
    """
    # --- 1. Load Configuration ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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
    print("\nEvaluation finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run evaluation on a trained NER or RE model.")
    
    parser.add_argument(
        '--config-path', 
        type=str, 
        required=True, 
        help='Path to the YAML configuration file for evaluation (e.g., `configs/evaluation_ner_config.yaml`).'
    )
    
    args = parser.parse_args()
    
    run_evaluation(args.config_path)