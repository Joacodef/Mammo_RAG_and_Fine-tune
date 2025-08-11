# scripts/run_ner_training.py
import argparse
import yaml
from pathlib import Path
import os
import torch
import sys

# Add the project root to the Python path to allow for absolute imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader.ner_datamodule import NERDataModule
from src.models.bert_ner import BertNerModel
from src.training.trainer import Trainer

def run_batch_training(config_path, partition_dir):
    """
    Main function to run the training process for all samples in a partition directory.

    Args:
        config_path (str): Path to the main training YAML configuration file.
        partition_dir (str): Path to the directory containing training samples
                             (e.g., 'data/processed/train-50').
    """
    # --- 1. Load Configuration and Find Samples ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    base_partition_dir = Path(partition_dir)
    sample_dirs = sorted([d for d in base_partition_dir.iterdir() if d.is_dir() and d.name.startswith('sample-')])

    if not sample_dirs:
        print(f"Warning: No sample directories found in '{partition_dir}'. Exiting.")
        return

    print(f"Found {len(sample_dirs)} samples to process in '{base_partition_dir.name}'.")

    # --- 2. Loop Through Each Sample and Train a Model ---
    for sample_dir in sample_dirs:
        train_file_path = sample_dir / "train.jsonl"
        if not train_file_path.exists():
            print(f"  - Skipping {sample_dir.name}: 'train.jsonl' not found.")
            continue
        
        print(f"\n{'='*20} Starting Training for: {sample_dir.name} {'='*20}")

        # Set the seed for reproducibility for each run
        torch.manual_seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed'])

        # --- Initialize Data Module ---
        print(f"Loading data from: {train_file_path}")
        datamodule = NERDataModule(config=config, train_file=train_file_path)
        datamodule.setup()
        
        n_labels = len(datamodule.label_map)
        print(f"Number of labels in the dataset: {n_labels}")

        # --- Initialize Model ---
        print(f"Initializing model: {config['model']['base_model']}")
        model = BertNerModel(
            base_model=config['model']['base_model'],
            n_labels=n_labels
        )

        # --- Initialize Trainer ---
        print("Initializing trainer...")
        trainer = Trainer(
            model=model,
            datamodule=datamodule,
            config=config
        )

        # --- Start Training ---
        trainer.train()

        # --- Save the final model ---
        base_output_dir = Path(config['paths']['output_dir'])
        model_name = config['model']['base_model'].replace("/", "_")
        data_partition_name = base_partition_dir.name
        sample_name = sample_dir.name
        
        final_output_dir = base_output_dir / model_name / data_partition_name / sample_name
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving final model to: {final_output_dir}")
        trainer.save_model(final_output_dir)
        
        print(f"\n{'='*20} Finished Training for: {sample_dir.name} {'='*20}")

    print("\nAll training experiments finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a batch of NER training experiments for all samples within a given data partition."
    )
    
    parser.add_argument(
        '--config-path', 
        type=str, 
        required=True, 
        help='Path to the YAML configuration file for training.'
    )
    
    parser.add_argument(
        '--partition-dir',
        type=str,
        required=True,
        help="Path to the directory containing the training samples (e.g., 'data/processed/train-50')."
    )
    
    args = parser.parse_args()
    
    run_batch_training(args.config_path, args.partition_dir)