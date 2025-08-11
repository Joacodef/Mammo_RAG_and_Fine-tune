# Comparison of RAG and Fine-Tuning for Mammogram Report NER

This repository contains the code for a project that compares two methodologies for Named Entity Recognition (NER) on mammogram reports:

1.  **Fine-tuning** traditional transformer models (e.g., BERT) on a labeled dataset.
2.  **Few-shot prompting** with Large Language Models (LLMs) like GPT via a Retrieval-Augmented Generation (RAG) pipeline.

The primary objective is to evaluate how the performance of these two approaches scales with the amount of available labeled data, from very few examples (e.g., 5-10 reports) to a larger corpus.

## Key Features

  - **Modular Model Framework**: Allows for easy substitution between locally fine-tuned models and API-based RAG models through a unified interface.
  - **Configuration-Driven Experiments**: Ensures reproducibility and simplifies experiment management using YAML configuration files.
  - **RAG Pipeline**: Implements components for text embedding, vector storage (using FAISS), and efficient similarity search for few-shot prompt creation.
  - **Reproducible Data Sampling**: Includes a script to automatically generate multiple, distinct data samples for various training set sizes (e.g., 5, 10, 50, 100+).
  - **Standardized Evaluation**: Calculates and reports key NER metrics, including entity-level $F\_1$-score, Precision, and Recall.

## Project Structure

The repository is organized to maintain a clear separation between configuration, source code, data, and results.

```
.
├── configs/                  # Experiment configuration files
│   └── data_config.yaml
├── data/                     # (Git-ignored) Raw and processed data
│   ├── raw/                  # Place raw train.jsonl here
│   └── processed/
│       └── train-5/
│           ├── sample-1/
│           │   └── train.jsonl
│           └── ...
├── notebooks/                # Jupyter notebooks for exploration and analysis
├── output/                   # (Git-ignored) Models, logs, vector stores, etc.
├── scripts/                  # High-level scripts to run experiments
│   └── data/
│       ├── generate_partitions.py
│       └── README.md
├── src/                      # Source code for the project
│   ├── data_loader/
│   ├── models/
│   ├── rag/
│   ├── training/
│   └── utils/
├── .env_example              # Example environment variables file
├── .gitignore
└── requirements.txt
```

## Setup and Installation

Follow these steps to configure the project environment.

### Prerequisites

  - Python 3.11+
  - A virtual environment manager, such as `venv` or `conda`.

### Installation Steps

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd Mammo_RAG_and_Fine-tune
    ```

2.  **Create and activate a virtual environment:**
    For example, using `conda`:

    ```bash
    conda create --name mammo-ner python=3.11
    conda activate mammo-ner
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    If you plan to use API-based models, copy the example environment file.

    ```bash
    cp .env_example .env
    ```

    Open the newly created `.env` file and add your secret keys (e.g., `OPENAI_API_KEY=...`).

## Data Setup: Generating Training Samples

This project's experiments rely on training data samples of various sizes, which are generated from a single raw `.jsonl` file. The process is handled by a configurable script that creates multiple, reproducible samples for each desired data size.

**For detailed instructions on how to configure and run the data generation script, please see the dedicated documentation:**

➡️ **[Data Generation README](./scripts/data/README.md)**
