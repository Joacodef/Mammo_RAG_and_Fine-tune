# scripts/data/generate_partitions.py
import json
import os
import random
import argparse
import yaml
from pathlib import Path

def read_jsonl(file_path):
    """
    Reads a .jsonl file and returns a list of dictionaries.

    Args:
        file_path (str or Path): The path to the .jsonl file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a line in the file.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """
    Saves a list of dictionaries to a .jsonl file.

    Args:
        data (list): A list of dictionaries to save.
        file_path (str or Path): The path to the output .jsonl file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def remove_comments(data):
    """
    Removes the 'Comments' key from each record in the dataset.

    Args:
        data (list): A list of data records (dictionaries).

    Returns:
        list: The cleaned list of data records.
    """
    for record in data:
        if 'Comments' in record:
            del record['Comments']
    return data

def generate_partitions(config):
    """
    Generates training and validation partitions based on the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters.
    """
    # Load configuration parameters
    base_dir = Path().cwd()
    input_file = base_dir / config['data']['input_file']
    output_dir_base = base_dir / config['data']['output_dir']
    partition_sizes = config['data']['partition_sizes']
    n_samples = config['data']['n_samples']
    base_seed = config['data']['base_seed']

    # Read and preprocess the raw data
    print(f"Reading raw data from: {input_file}")
    raw_data = read_jsonl(input_file)
    cleaned_data = remove_comments(raw_data)
    print(f"Successfully loaded and cleaned {len(cleaned_data)} records.")

    # Generate each partition sample
    for size in partition_sizes:
        print(f"\nGenerating {n_samples} samples for partition size: '{size}'")

        # Create n_samples for the current partition size
        for i in range(n_samples):
            sample_num = i + 1
            # Calculate the seed for the current sample to ensure reproducibility
            current_seed = base_seed + i

            # Shuffle the full dataset with the calculated seed
            data_to_shuffle = list(cleaned_data)
            random.seed(current_seed)
            random.shuffle(data_to_shuffle)

            # Determine the number of records for the training set
            if size == "all":
                train_size = len(data_to_shuffle)
            else:
                train_size = size

            # Ensure there's enough data
            if train_size > len(data_to_shuffle):
                print(f"  - Warning: Not enough data for sample {sample_num} (size {train_size}). Skipping.")
                continue

            # Take the top N records for the training sample
            train_data = data_to_shuffle[:train_size]

            # Define output directory for the current sample
            output_dir_sample = output_dir_base / f"train-{size}" / f"sample-{sample_num}"
            output_dir_sample.mkdir(parents=True, exist_ok=True)

            # Save the training data
            train_output_path = output_dir_sample / "train.jsonl"
            save_jsonl(train_data, train_output_path)
            
            print(f"  - Sample {sample_num} (seed {current_seed}): Train={len(train_data)}. Saved to: {output_dir_sample}")

    print("\nData sampling complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate data partitions based on a YAML configuration file.")
    parser.add_argument('--config-path', type=str, required=True, help='Path to the YAML configuration file.')
    
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    generate_partitions(config)