# Training Scripts

## Overview

This directory contains the high-level scripts used to launch training experiments for both **Named Entity Recognition (NER)** and **Relation Extraction (RE)** models. These scripts are the primary entry points for the training workflow, designed to be run from the command line.

They are built to handle **batch training**, meaning they can automatically find and process multiple training data samples within a given partition directory (e.g., all 5 samples in `data/processed/train-50/`), training a separate model for each one. This is essential for the project's goal of evaluating model performance across different data subsets.

-----

## `run_ner_training.py`

### Purpose

This script manages the end-to-end training process for NER models. It initializes the necessary components (`NERDataModule`, `BertNerModel`, `Trainer`), orchestrates the training loop, and saves the final model artifacts.

### Workflow

1.  **Configure Training**: Ensure the `configs/training_ner_config.yaml` file is set up with the desired parameters (e.g., base model, learning rate, number of epochs).

2.  **Execute the Script**: Run the script from the **root directory**, specifying the path to the NER training configuration and the target partition directory that contains the training samples.

    ```bash
    python scripts/training/run_ner_training.py \
      --config-path configs/training_ner_config.yaml \
      --partition-dir data/processed/train-50
    ```

3.  **Output**: The script will create a unique, timestamped directory for the entire run. Each trained model sample will be saved within this directory. For example: `output/models/ner/train-50/20240828_120400/sample-1/`, containing the model weights, tokenizer files, and a copy of the training configuration.

-----

## `run_re_training.py`

### Purpose

This script manages the training process for RE models. It is structurally similar to the NER script but is tailored for relation extraction, initializing the `REDataModule` and `REModel`.

### Workflow

1.  **Configure Training**: Ensure the `configs/training_re_config.yaml` file is configured for the RE task.

2.  **Execute the Script**: Run the script from the **root directory**, pointing to the RE configuration and the desired data partition. I recommend renaming the script file from `run_re_trainig.py` to `run_re_training.py` to fix the typo.

    ```bash
    python scripts/training/run_re_training.py \
      --config-path configs/training_re_config.yaml \
      --partition-dir data/processed/train-50
    ```

3.  **Output**: The script creates a unique, timestamped directory for the entire run, inside which each trained model sample will be saved. This prevents accidental overwriting of previous results. For example: `output/models_re/re/train-50/20240828_120729/sample-1/`.