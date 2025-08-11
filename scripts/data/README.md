# Data Generation Scripts

## Overview

This directory contains scripts for preparing and sampling data for experiments. The primary script is `generate_partitions.py`, which is designed to create a consistent, stratified holdout test set and multiple, reproducible training samples of various sizes from a single raw dataset.

## Script: `generate_partitions.py`

### Purpose

This script automates the creation of data partitions for robust model evaluation. It performs two key functions:
1.  **Global Test Set Creation**: It first splits the entire raw dataset into a global training set and a holdout test set using multi-label stratification. This ensures that all models, regardless of the training data size, are evaluated against the same, representative test data.
2.  **Stratified Training Samples**: From the global training set, it generates multiple, smaller training partitions of various sizes (e.g., 5, 10, 50 reports). This is essential for evaluating model performance at different data scales.

The entire process is controlled by a single YAML configuration file to ensure reproducibility.

### Workflow

The data generation process follows these steps:

1.  **Provide Raw Data**: Before running the script, place your full, labeled dataset in the `data/raw/` directory. The script expects the file to be named `all.jsonl` by default, but this can be changed in the configuration file.

2.  **Expected Data Format**: The input file must be in the **JSON Lines (`.jsonl`)** format, where each line is a self-contained, valid JSON object. Each object must contain a `"text"` key and an `"entities"` key. The script will automatically remove the optional `"Comments"` key if it exists.

    **Example of a single line in spanish (JSON object):**

    ```json
    {
      "id": "1.2.840.113619.2.373.202306131802469930109650",
      "text": "Ambas mamas son densas y heterogéneas.\nMicrocalcificaciones aisladas.\nNódulo periareolar derecho bien delimitadp de 10mm.\nNódulo calcifcado derecho.\nNo observo microcalcificaciones sospechosas agrupadas ni imágenes espiculadas.\nRegiones axilares sin adenopatías.\nImpresión: Mamas densas y nódulo derecho presuntamente benigno.\nSugiero ecografía mamaria.\nBI-RADS 3 ACR C",
      "Comments": [],
      "entities": [
        {"id": 4762, "label": "DENS", "start_offset": 0, "end_offset": 37},
        {"id": 4763, "label": "HALL_presente", "start_offset": 39, "end_offset": 59},
        {"id": 4765, "label": "HALL_presente", "start_offset": 70, "end_offset": 76},
        {"id": 4769, "label": "HALL_presente", "start_offset": 289, "end_offset": 295}
      ],
      "relations": [
        {"id": 22, "from_id": 4764, "to_id": 4763, "type": "describir"},
        {"id": 29, "from_id": 4770, "to_id": 4765, "type": "ubicar"},
        {"id": 5254, "from_id": 13447, "to_id": 13446, "type": "ubicar"}
      ]
    }
    ```

3.  **Configure Data Generation**: The script's behavior is controlled by `configs/data_preparation_config.yaml`. Open this file to define the parameters for splitting and sampling.

    **Example `configs/data_preparation_config.yaml`:**

    ```yaml
    data:
      base_seed: 42
      n_samples: 5
      test_split_ratio: 0.2
      raw_input_file: 'data/raw/all.jsonl'
      partitions_dir: 'data/processed'
      holdout_test_set_path: 'data/processed/test.jsonl'
      partition_sizes: [5, 10, 20, 50, 100, "all"]
    ```

4.  **Execute the Script**: Run the script from the **root directory** of the project, providing the path to the configuration file.

    ```bash
    python scripts/data/generate_partitions.py --config-path configs/data_preparation_config.yaml
    ```

5.  **Review the Output**: The script will generate a holdout test file and a nested directory structure within the specified `partitions_dir` (`data/processed/`). The script will also print a label distribution report for the global train and test sets to the console.

    **Example Output Structure:**

    ```
    data/processed/
    ├── test.jsonl              # Global holdout test set
    ├── train-50/
    │   ├── sample-1/
    │   │   └── train.jsonl
    │   ├── sample-2/
    │   │   └── train.jsonl
    │   └── ...
    └── train-100/
        ├── sample-1/
        │   └── train.jsonl
        └── ...
    ```