import argparse
import yaml
from pathlib import Path
import json
import torch
from seqeval.metrics import classification_report

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader.ner_datamodule import NERDataModule
from src.evaluation.predictor import Predictor

def run_evaluation(config_path):
    """
    Main function to run the evaluation on a test set.

    Args:
        config_path (str): Path to the evaluation YAML configuration file.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_path = config['model_path']
    test_file = config['test_file']
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Initialize Data Module ---
    print(f"Loading test data from: {test_file}")
    # We need entity labels for mapping, which should be part of the model's config
    # or specified in the evaluation config.
    datamodule = NERDataModule(config=config, test_file=test_file)
    datamodule.setup()
    test_loader = datamodule.test_dataloader()

    # --- 2. Initialize Predictor ---
    print(f"Loading model from: {model_path}")
    predictor = Predictor(model_path=model_path, config=config)

    # --- 3. Get Predictions ---
    predictions, true_labels = predictor.predict(test_loader)

    # --- 4. Convert IDs back to Labels for Evaluation ---
    # Create an inverted label map to convert label IDs back to string tags (e.g., "B-FIND")
    inv_label_map = {v: k for k, v in datamodule.label_map.items()}
    
    true_labels_str = [[inv_label_map[l] for l in seq] for seq in true_labels]
    pred_labels_str = [[inv_label_map[p] for p, l in zip(pred_seq, true_seq)] 
                       for pred_seq, true_seq in zip(predictions, true_labels)]

    # --- 5. Calculate and Save Metrics ---
    print("\n--- Evaluation Results ---")
    report = classification_report(true_labels_str, pred_labels_str, output_dict=True)
    
    # Print a user-friendly version of the report
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"\nEntity: {key}")
            for metric, score in value.items():
                print(f"  {metric:<10}: {score:.4f}")
        else:
            print(f"\n{key:<20}: {value:.4f}")
            
    # Save the detailed report to a JSON file
    report_path = output_dir / f"evaluation_metrics_{Path(model_path).name}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    print(f"\nEvaluation metrics saved to: {report_path}")
    print("\nEvaluation finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run evaluation on a trained NER model.")
    
    parser.add_argument(
        '--config-path', 
        type=str, 
        required=True, 
        help='Path to the YAML configuration file for evaluation.'
    )
    
    args = parser.parse_args()
    
    run_evaluation(args.config_path)