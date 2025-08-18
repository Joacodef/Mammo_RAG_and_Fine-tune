import argparse
import yaml
from pathlib import Path
import json
import torch
import shutil
from datetime import datetime
import numpy as np

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

def run_prediction_and_save(config):
    """
    Main function to run predictions on a test set and save the raw outputs.
    This function no longer calculates metrics.
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
        print("Initializing NER components for prediction...")
        datamodule = NERDataModule(config=config, test_file=test_file)
    else: # task == 're'
        print("Initializing RE components for prediction...")
        datamodule = REDataModule(config=config, test_file=test_file)
    
    datamodule.setup()
    test_loader = datamodule.test_dataloader()

    # --- 3. Initialize Model and Predictor ---
    if task == 'ner':
        model = BertNerModel(base_model=model_path)
    else: # task == 're'
        model = REModel(base_model=model_path, tokenizer=datamodule.tokenizer)

    print(f"Loading model from: {model_path}")
    predictor = Predictor(model=model, device=device)

    # --- 4. Get Raw Predictions ---
    predictions, true_labels, _ = predictor.predict(test_loader, task_type=task)

    # --- 5. Save Raw Outputs ---
    # We also need the original text to provide full context in the output file.
    # We load it directly from the test file.
    with open(test_file, 'r', encoding='utf-8') as f:
        source_records = [json.loads(line) for line in f]

    output_data = []
    for i, record in enumerate(source_records):
        output_data.append(convert_numpy_types({
            "source_text": record.get("text", ""),
            "true_labels": true_labels[i],
            "predicted_labels": predictions[i]
        }))

    # Save the raw predictions to a file
    output_filename = output_dir / f"raw_predictions_{Path(model_path).name}.jsonl"
    with open(output_filename, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')

    print(f"\nRaw predictions saved to: {output_filename}")
    print(f"Prediction generation for '{Path(model_path).name}' finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a batch of predictions for all models in a given directory and save raw outputs."
    )
    
    parser.add_argument(
        '--config-path', 
        type=str, 
        required=True, 
        help='Path to the YAML configuration file for evaluation.'
    )
    
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    model_dir = Path(base_config['model_dir'])
    sample_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('sample-')])

    if not sample_dirs:
        raise FileNotFoundError(f"No 'sample-*' directories found in '{model_dir}'.")

    # Create a unique, timestamped output directory for this evaluation run
    train_size_folder_name = model_dir.parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_timestamped = Path(base_config['output_dir']) / train_size_folder_name / timestamp
    output_dir_timestamped.mkdir(parents=True, exist_ok=True)
    
    print(f"All prediction outputs for this run will be saved in: {output_dir_timestamped}")
    shutil.copy(args.config_path, output_dir_timestamped / "evaluation_config.yaml")

    print(f"Found {len(sample_dirs)} model samples to generate predictions for.")

    for i, sample_path in enumerate(sample_dirs):
        print(f"\n{'='*20} Generating predictions for: {sample_path.name} ({i+1}/{len(sample_dirs)}) {'='*20}")
        
        sample_config = base_config.copy()
        sample_config['model_path'] = str(sample_path)
        sample_config['output_dir'] = str(output_dir_timestamped)
        
        run_prediction_and_save(sample_config)

    print("\nBatch prediction generation finished successfully.")