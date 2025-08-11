import argparse
import yaml
from pathlib import Path
import os
import torch

# Add the project root to the Python path to allow for absolute imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader.ner_datamodule import NERDataModule
from src.models.bert_ner import BertNerModel
from src.training.trainer import Trainer

def run_training(config_path):
    """
    Main function to run the training process for a single experiment.

    Args:
        config_path (str): Path to the main training YAML configuration file.
    """
    # Load configuration from YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get train_file_path from the config
    train_file_path = config['paths']['train_file'] 

    # Set the seed for reproducibility
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # --- 1. Initialize Data Module ---
    print(f"Loading data from: {train_file_path}")
    datamodule = NERDataModule(config=config, train_file=train_file_path)
    datamodule.setup()
    
    # Dynamically determine the number of labels from the datamodule's label_map
    n_labels = len(datamodule.label_map)
    print(f"Number of labels in the dataset: {n_labels}")


    # --- 2. Initialize Model ---
    print(f"Initializing model: {config['model']['base_model']}")
    model = BertNerModel(
        base_model=config['model']['base_model'],
        n_labels=n_labels
    )

    # --- 3. Initialize Trainer ---
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        datamodule=datamodule,
        config=config
    )

    # --- 4. Start Training ---
    trainer.train()

    # --- 5. Save the final model ---
    # Create a unique output directory for this run
    # Example: output/models/bert-base-cased/train-50/sample-1
    base_output_dir = Path(config['paths']['output_dir'])
    model_name = config['model']['base_model'].replace("/", "_") # Handle model names with slashes
    data_partition_name = Path(train_file_path).parent.parent.name # e.g., "train-50"
    sample_name = Path(train_file_path).parent.name # e.g., "sample-1"
    
    final_output_dir = base_output_dir / model_name / data_partition_name / sample_name
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving final model to: {final_output_dir}")
    trainer.save_model(final_output_dir)
    
    print("\nExperiment finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a training experiment for NER.")
    
    parser.add_argument(
        '--config-path', 
        type=str, 
        required=True, 
        help='Path to the YAML configuration file for training.'
    )
    
    args = parser.parse_args()
    
    run_training(args.config_path)