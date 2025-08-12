# Evaluation Module

## Overview

This directory contains the script responsible for running model inference and generating predictions for both **Named Entity Recognition (NER)** and **Relation Extraction (RE)** tasks. The primary goal of this module is to provide a standardized, task-agnostic interface for evaluating a trained model on a test dataset.

---

## `predictor.py`

### Purpose

The `Predictor` class encapsulates the logic for loading a fine-tuned model, running it in evaluation mode, and processing the model's output logits to produce clean, usable predictions. It is designed to work seamlessly with the `DataLoader` instances provided by the `NERDataModule` or `REDataModule`.

### Key Features

-   **Task-Agnostic Design**: A single `predict` method handles both NER and RE tasks. It dynamically adapts its prediction logic based on a `task_type` parameter, correctly interpreting either token-level (NER) or sequence-level (RE) logits.
-   **Device Management**: Automatically moves the model and data batches to the appropriate device (`cuda` or `cpu`), ensuring efficient inference.
-   **Batch Processing**: Iterates through a `DataLoader`, processes batches of data, and accumulates predictions and true labels.
-   **Label Alignment (NER)**: For NER tasks, it correctly aligns the predicted token labels with the true labels, filtering out any padded tokens (identified by the label `-100`) to ensure that metrics are calculated only on the actual sequence content.
-   **Clean Output**: Returns two simple lists—one for all predictions and one for all corresponding true labels—which can be directly consumed by standard evaluation libraries like `seqeval` or `scikit-learn`.