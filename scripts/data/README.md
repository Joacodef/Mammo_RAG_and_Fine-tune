
# Data Generation Scripts

## Overview

This directory contains scripts for preparing and sampling data for experiments. The primary script is `generate_partitions.py`, which is designed to create multiple, reproducible training samples of various sizes from a raw dataset.

## Script: `generate_partitions.py`

### Purpose

This script automates the creation of training data subsets. It reads a single raw `.jsonl` file and generates a structured output of smaller datasets, which is essential for evaluating model performance at different data scales. The process is controlled entirely by a YAML configuration file to ensure reproducibility.

### Workflow

The data generation process follows these steps:

1.  **Provide Raw Data**: Before running the script, place your full, labeled dataset in the `data/raw/` directory. The script expects this file to be named `train.jsonl`.

2.  **Expected Data Format**: The input file must be in the **JSON Lines (`.jsonl`)** format, where each line is a self-contained, valid JSON object. Each object should contain at least a `"text"` key and an `"entities"` key. The script is also designed to automatically remove the optional `"Comments"` key if it exists.

    **Example of a single line (JSON object):**

    ```json
    {
      "id": 123,
      "text": "Report content...",
      "entities": [
        {"id": 0, "start_offset": 12, "end_offset": 29, "label": "FIND"},
        {"id": 1, "start_offset": 36, "end_offset": 48, "label": "REG"}
      ],
      "relations": [],
      "Comments": "This is an optional comment that will be removed."
    }
    ```

3.  **Configure Data Generation**: The script's behavior is controlled by `configs/data_config.yaml`. Open this file to define the parameters for sampling.

    **Example `configs/data_config.yaml`:**

    ```yaml
    data:
      base_seed: 42
      n_samples: 5
      input_file: 'data/raw/train.jsonl'
      output_dir: 'data/processed'
      partition_sizes: [5, 10, 20, 50, 100, "all"]
    ```

4.  **Execute the Script**: Run the script from the **root directory** of the project, providing the path to the configuration file.

    ```bash
    python scripts/data/generate_partitions.py --config-path configs/data_config.yaml
    ```

5.  **Review the Output**: The script will generate a nested directory structure within the specified `output_dir` (`data/processed/`). For each partition size, it will create a directory containing subdirectories for each sample.

    **Example Output Structure:**

    ```
    data/processed/
    └── train-50/
        ├── sample-1/
        │   └── train.jsonl
        ├── sample-2/
        │   └── train.jsonl
        └── ...
    ```