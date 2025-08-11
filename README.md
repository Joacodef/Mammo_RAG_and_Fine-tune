# Comparison of RAG and Fine-Tuning for Mammogram Report Analysis

This repository contains the code for a project that compares two methodologies for Natural Language Processing (NLP) tasks on mammogram reports:

1.  **Fine-tuning** traditional transformer models (e.g., BERT) on a labeled dataset for both **Named Entity Recognition (NER)** and **Relation Extraction (RE)**.
2.  **Few-shot prompting** with Large Language Models (LLMs) like GPT via a Retrieval-Augmented Generation (RAG) pipeline for NER.

The primary objective is to evaluate how the performance of these two approaches scales with the amount of available labeled data, from very few examples (e.g., 5-10 reports) to a larger corpus.

-----

## Key Features

  - **Dual-Task Support**: Implements distinct, fine-tunable models for both **NER** and **Relation Extraction (RE)**, allowing for comprehensive information extraction.
  - **Modular Model Framework**: Allows for easy substitution between locally fine-tuned models and API-based RAG models through a unified interface.
  - **Configuration-Driven Experiments**: Ensures reproducibility and simplifies experiment management for both training and evaluation using YAML configuration files.
  - **Reproducible Data Sampling**: Includes a script to automatically generate multiple, distinct, and stratified data samples for various training set sizes, ensuring balanced label distribution.
  - **Standardized Evaluation**: Calculates and reports key metrics for both NER (entity-level $F\_1$-score, Precision, Recall using `seqeval`) and RE (classification reports using `scikit-learn`).
  - **Automated Testing**: Integrated with GitHub Actions for continuous integration, running a full suite of unit tests with `pytest` on every push and pull request to the main branch.

-----

## Project Structure

The repository is organized to maintain a clear separation between configuration, source code, data, and results.

```
.
├── .github/workflows/        # CI workflows for automated testing
│   └── python-tests.yml
├── configs/                  # Experiment configuration files
│   ├── data_preparation_config.yaml
│   ├── training_ner_config.yaml
│   ├── training_re_config.yaml
│   ├── evaluation_ner_config.yaml
│   └── evaluation_re_config.yaml
├── data/                     # (Git-ignored) Raw and processed data
│   ├── raw/                  # Place raw all.jsonl here
│   └── processed/
│       ├── test.jsonl
│       └── train-5/
│           ├── sample-1/
│           │   └── train.jsonl
│           └── ...
├── output/                   # (Git-ignored) Models, logs, evaluation results, etc.
├── scripts/                  # High-level scripts to run experiments
│   ├── data/
│   │   └── generate_partitions.py
│   ├── run_ner_training.py
│   ├── run_re_training.py
│   └── run_evaluation.py
├── src/                      # Source code for the project
│   ├── data_loader/          # NER and RE dataloaders
│   ├── models/               # NER and RE model definitions
│   ├── evaluation/           # Prediction and evaluation logic
│   └── training/             # Reusable training loop
├── tests/                    # Unit and integration tests
│   └── unit/
├── .gitignore
├── requirements.txt
└── README.md
```

-----

## Setup and Installation

Follow these steps to configure the project environment.

### Prerequisites

  - Python 3.11+
  - A virtual environment manager, such as `venv` or `conda`.

### Installation Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Joacodef/Mammo_RAG_and_Fine-tune.git
    cd Mammo_RAG_and_Fine-tune
    ```

2.  **Create and activate a virtual environment:**
    For example, using `conda`:

    ```bash
    conda create --name mammo-nlp python=3.11
    conda activate mammo-nlp
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    If you plan to use API-based models (e.g., OpenAI), copy the example environment file.

    ```bash
    cp .env_example .env
    ```

    Open the newly created `.env` file and add your secret keys (e.g., `OPENAI_API_KEY=...`).

-----

## Running the Experiments

The entire experimental workflow, from data preparation to evaluation, is executed using command-line scripts.

### 1\. Generate Data Partitions

First, generate the stratified training and test sets from your raw data file. Ensure your data is located at `data/raw/all.jsonl` and configure the partitions in `configs/data_preparation_config.yaml`.

```bash
python scripts/data/generate_partitions.py --config-path configs/data_preparation_config.yaml
```

**For more details, see the [Data Generation README](https://www.google.com/search?q=./scripts/data/README.md)**.

### 2\. Train Models

Train either NER or RE models across the generated data partitions. The script will iterate through each sample in the specified partition directory (e.g., `data/processed/train-50`).

  - **To run NER training:**

    ```bash
    python scripts/run_ner_training.py \
      --config-path configs/training_ner_config.yaml \
      --partition-dir data/processed/train-50
    ```

  - **To run RE training:**

    ```bash
    python scripts/run_re_training.py \
      --config-path configs/training_re_config.yaml \
      --partition-dir data/processed/train-50
    ```

### 3\. Evaluate Models

After training, run evaluation on the holdout test set. Update the `model_path` in the corresponding evaluation config file (`configs/evaluation_ner_config.yaml` or `configs/evaluation_re_config.yaml`) to point to the trained model you wish to evaluate.

  - **To run NER evaluation:**

    ```bash
    python scripts/run_evaluation.py --config-path configs/evaluation_ner_config.yaml
    ```

  - **To run RE evaluation:**

    ```bash
    python scripts/run_evaluation.py --config-path configs/evaluation_re_config.yaml
    ```

-----

## Testing

The repository includes a suite of unit tests to ensure code quality and correctness. To run the tests locally, execute the following command from the root directory:

```bash
python -m pytest
```

Tests are also run automatically via GitHub Actions on every push or pull request to the `main` branch.