# Evaluation Scripts

## Overview

This directory contains the scripts for running the model evaluation pipeline. The process is divided into two main stages:

1.  **Prediction Generation**: First, raw predictions are generated for a given model on the test set. This is handled by separate scripts for fine-tuned models and the RAG pipeline. The output is a `.jsonl` file containing the source text, true labels, and predicted labels for each record.
2.  **Metric Calculation**: Once the raw prediction file is generated, a second script is used to calculate the final performance metrics (e.g., precision, recall, $F\_1$-score) and save them to a structured JSON report.

This two-step approach decouples the often time-consuming inference process from the rapid calculation of metrics, allowing for more flexible analysis.

-----

## Workflow

The standard evaluation workflow is as follows:

1.  **Generate Predictions**: Run the appropriate script to generate a raw predictions file for your model.
      * For fine-tuned BERT models (NER or RE), use `generate_finetuned_predictions.py`.
      * For the RAG-based LLM, use `generate_rag_predictions.py`.
2.  **Calculate Metrics**: Run the `calculate_final_metrics.py` script, pointing it to the prediction file generated in the previous step to get the final, quantitative results.

-----

## `generate_finetuned_predictions.py`

### Purpose

This script runs inference for fine-tuned **NER** and **RE** models on the test set and saves the raw, unprocessed predictions. It is designed to process all model "samples" within a specific training partition directory in a single run (e.g., all 5 models trained on the `train-50` partition).

### Usage

The script is driven by a YAML configuration file (`configs/evaluation_ner_config.yaml` or `configs/evaluation_re_config.yaml`) which specifies the model directory, test data, and other parameters.

```bash
python scripts/evaluation/generate_finetuned_predictions.py \
  --config-path configs/evaluation_ner_config.yaml
```

-----

## `generate_rag_predictions.py`

### Purpose

This script runs the full end-to-end RAG pipeline to generate NER predictions for the test set. It handles vector database queries, prompt formatting, LLM calls, and saves the final extracted entities to a `.jsonl` file.

### Usage

The script is configured using `configs/rag_config.yaml`.

```bash
python scripts/evaluation/generate_rag_predictions.py \
  --config-path configs/rag_config.yaml
```

-----

## `calculate_final_metrics.py`

### Purpose

This script takes a raw prediction file (or a directory of prediction files) and calculates the final performance metrics. It can process a single file for a detailed report or an entire directory to generate aggregate statistics (mean and standard deviation) across multiple samples.

### Usage

You must specify the type of task evaluation (`ner`, `rag`, or `re`) and provide either a path to a single prediction file or a directory. Note that the `--config-path` argument is only required for the `re` (Relation Extraction) task type.

  * **To calculate metrics for a single fine-tuned NER model's predictions:**

    ```bash
    python scripts/evaluation/calculate_final_metrics.py \
      --prediction-path output/finetuned_results/ner/train-50/20240828_150000/predictions_sample-1.jsonl \
      --type ner \
      --output-path output/finetuned_results/ner/train-50/20240828_150000/final_metrics_sample-1.json
    ```

  * **To calculate aggregate metrics for a batch of fine-tuned models:**
    This is the primary method for getting the final averaged performance for an experiment.

    ```bash
    python scripts/evaluation/calculate_final_metrics.py \
      --prediction-dir output/finetuned_results/ner/train-50/20240828_150000/ \
      --type ner \
      --output-path output/finetuned_results/ner/train-50/20240828_150000/aggregate_metrics_report.json
    ```

  * **To calculate metrics for RAG predictions:**

    ```bash
    python scripts/evaluation/calculate_final_metrics.py \
      --prediction-path output/rag_results/ner/20240828_160000/predictions.jsonl \
      --type rag \
      --output-path output/rag_results/ner/20240828_160000/final_metrics.json
    ```