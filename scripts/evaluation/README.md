# Evaluation Scripts

## Overview

This directory contains the primary script for evaluating the performance of fine-tuned **Named Entity Recognition (NER)** and **Relation Extraction (RE)** models. The script is designed to be executed from the command line and relies on a YAML configuration file to specify the model, data, and other parameters for the evaluation run.

-----

## `run_evaluation.py`

### Purpose

The `run_evaluation.py` script orchestrates the entire evaluation process. It loads a trained model, runs it on a specified test set, calculates the relevant performance metrics, and saves the results to a file. It is designed to be flexible and can evaluate either NER or RE models based on the provided configuration.

### Workflow

1.  **Configure the Evaluation**: Before running the script, you must prepare a YAML configuration file (e.g., `configs/evaluation_ner_config.yaml` or `configs/evaluation_re_config.yaml`). This file tells the script which task to run (`ner` or `re`), where to find the trained model, which test file to use, and where to save the output.

    **Example `configs/evaluation_ner_config.yaml`:**

    ```yaml
    # Task type to guide the evaluation script
    task: "ner"

    # Path to the directory containing the fine-tuned NER model.
    model_path: 'output/models/bert-base-cased/train-50/sample-1'

    # Path to the test data file (.jsonl format).
    test_file: 'data/processed/test.jsonl'

    # A list of the distinct entity types for the NER model.
    entity_labels:
      - "FIND"
      - "REG"

    # Directory where the evaluation results will be saved.
    output_dir: 'output/evaluation_results/ner'

    # Evaluation batch size.
    batch_size: 16
    ```

2.  **Execute the Script**: Run the script from the **root directory** of the project, providing the path to your evaluation configuration file using the `--config-path` argument.

      - **To run NER evaluation:**

        ```bash
        python scripts/evaluation/run_evaluation.py --config-path configs/evaluation_ner_config.yaml
        ```

      - **To run RE evaluation:**

        ```bash
        python scripts/evaluation/run_evaluation.py --config-path configs/evaluation_re_config.yaml
        ```

3.  **Review the Output**: The script will print a detailed classification report to the console and save the same report as a JSON file in the specified `output_dir`. The output filename is automatically generated based on the name of the model being evaluated.

### Key Features

  - **Configuration-Driven**: All aspects of the evaluation are controlled by a single, clear YAML file, ensuring reproducibility.
  - **Task-Agnostic**: A single script can evaluate both NER and RE models by simply changing the `task` parameter in the configuration file.
  - **Standardized Metrics**:
      - For **NER**, it computes entity-level precision, recall, and $F\_1$-score using `seqeval.metrics.classification_report`.
      - For **RE**, it computes precision, recall, and $F\_1$-score for each relation class using `sklearn.metrics.classification_report`.
  - **Automated Results Saving**: Automatically saves a JSON file containing the full evaluation report, making it easy to compare results across different experiments.
  - **JSON Compatibility**: Includes a utility to convert any NumPy data types in the results to native Python types, ensuring the output report is always serializable to JSON.