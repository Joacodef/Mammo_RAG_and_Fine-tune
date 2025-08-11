# scripts/run_re_training.py
import argparse
import yaml
from pathlib import Path
import torch
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader.re_datamodule import REDataModule
from src.models.re_model import REModel
from src.training.trainer import Trainer

def run_re_training(config_path):
    """
    Main function to run the training process for a Relation Extraction experiment.

    Args:
        config_path (str): Path to the RE training YAML configuration file.
    """
    # Load configuration from YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_file_path = config['paths']['train_file']

    # Set the seed for reproducibility
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # --- 1. Initialize Data Module for RE ---
    print(f"Loading RE data from: {train_file_path}")
    datamodule = REDataModule(config=config, train_file=train_file_path)
    datamodule.setup()
    
    # Dynamically determine the number of relation labels
    n_labels = len(datamodule.relation_map)
    print(f"Number of relation labels in the dataset: {n_labels}")

    # --- 2. Initialize Model for RE ---
    print(f"Initializing RE model: {config['model']['base_model']}")
    # The RE model requires the tokenizer to be passed for resizing token embeddings
    model = REModel(
        base_model=config['model']['base_model'],
        n_labels=n_labels,
        tokenizer=datamodule.tokenizer
    )

    # --- 3. Initialize Trainer ---
    print("Initializing trainer...")
    # The generic Trainer class can be reused
    trainer = Trainer(
        model=model,
        datamodule=datamodule,
        config=config
    )

    # --- 4. Start Training ---
    trainer.train()

    # --- 5. Save the final model ---
    # Create a unique output directory for this RE run
    # Example: output/models_re/bert-base-cased/train-50/sample-1
    base_output_dir = Path(config['paths']['output_dir'])
    model_name = config['model']['base_model'].replace("/", "_")
    data_partition_name = Path(train_file_path).parent.parent.name
    sample_name = Path(train_file_path).parent.name
    
    final_output_dir = base_output_dir / model_name / data_partition_name / sample_name
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving final RE model to: {final_output_dir}")
    trainer.save_model(final_output_dir)
    
    print("\nRE experiment finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a training experiment for Relation Extraction.")
    
    parser.add_argument(
        '--config-path', 
        type=str, 
        required=True, 
        help='Path to the YAML configuration file for RE training.'
    )
    
    args = parser.parse_args()
    
    run_re_training(args.config_path)