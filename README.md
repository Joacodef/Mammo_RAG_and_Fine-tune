# Comparison of RAG and Fine-Tuning for Mammogram Report NER

This repository contains the code for a project that compares two methodologies for Named Entity Recognition (NER) on mammogram reports:
1.  **Fine-tuning** traditional transformer models (e.g., BERT) on a labeled dataset.
2.  **Few-shot prompting** with Large Language Models (LLMs) like GPT via a Retrieval-Augmented Generation (RAG) pipeline.

The primary objective is to evaluate how the performance of these two approaches scales with the amount of available labeled data, from very few examples (e.g., 5-10 reports) to a larger corpus.

## Key Features

- **Modular Model Framework**: Allows for easy substitution between locally fine-tuned models and API-based RAG models through a unified interface.
- **Configuration-Driven Experiments**: Ensures reproducibility and simplifies experiment management using YAML configuration files.
- **RAG Pipeline**: Implements components for text embedding, vector storage (using FAISS), and efficient similarity search for few-shot prompt creation.
- **Data Scaling Analysis**: Includes scripts to automatically generate data subsets for testing model performance at different data scales (5, 10, 20, 50, 100, 200+ samples).
- **Standardized Evaluation**: Calculates and reports key NER metrics, including entity-level $F_1$-score, Precision, and Recall.

## Project Structure

The repository is organized to maintain a clear separation between configuration, source code, data, and results, which is essential for reproducibility.

```
.
├── configs/                  # Experiment configuration files
├── data/                     # (Git-ignored) Raw, processed, and subset data
│   ├── full/
│   └── train-10/
│   └── ...
├── notebooks/                # Jupyter notebooks for exploration and analysis
├── output/                   # (Git-ignored) Models, logs, vector stores, results
├── scripts/                  # High-level scripts to run experiments
│   └── 01_prepare_data.py    # Script to create data subsets
│   └── 02_run_experiment.py  # Main experiment runner
├── src/                      # Source code for the project
│   ├── data_loader/          # Dataset classes and data processing
│   ├── models/               # Model wrappers (local vs. API)
│   ├── rag/                  # RAG components (vectorizer, vector store)
│   ├── training/             # Fine-tuning loop logic for local models
│   └── utils/                # Utility functions (config loading, metrics)
├── .env_example              # Example environment variables file
├── .gitignore
└── requirements.txt
```

## Setup and Installation

Follow these steps to configure the project environment.

### Prerequisites

- Python 3.9+
- A virtual environment manager, such as `venv` or `conda`.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd MAMMO_RAG_AND_FINE-TUNE
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    API keys for external services are managed via environment variables.
    ```bash
    # Copy the example file to create your own configuration
    cp .env_example .env
    ```
    Open the newly created `.env` file and populate it with the necessary secret keys (e.g., `OPENAI_API_KEY=...`). The application is configured to load these variables at runtime.

### Data Setup

1.  **Provide Raw Data**: Create a `data/raw/` directory and place the full dataset files (e.g., `train.json`, `val.json`, `test.json`) within it.

2.  **Generate Data Subsets**: Execute the data preparation script to create the smaller, sampled datasets required for the experiments.
    ```bash
    python scripts/01_prepare_data.py
    ```
    This script will populate the `data/` directory with `train-5`, `train-10`, and other subset directories.

## Running Experiments

Each experiment is defined and controlled by a dedicated configuration file in the `configs/` directory.

1.  **Define an Experiment Configuration**: Create a new configuration file by copying the base template.
    ```bash
    cp configs/experiment_base.yaml configs/bert_finetune_50_samples.yaml
    ```

2.  **Edit the Configuration**: Modify the new YAML file to specify the parameters for the experiment.
    ```yaml
    model_type: local
    model_name: 'bert-base-cased'
    data:
      train_subset_size: 50 # Specifies the 50-sample training set
      # ... other relevant parameters
    training_params:
      learning_rate: 5.0e-5
      epochs: 3
      # ... other relevant parameters
    ```

3.  **Execute the Experiment**: Run the main script, passing the path to your configuration file as an argument.
    ```bash
    python scripts/02_run_experiment.py --config configs/bert_finetune_50_samples.yaml
    ```

4.  **Review the Output**: The script will generate all artifacts, including model checkpoints, logs, and performance metrics, in a uniquely named folder within the `output/` directory.