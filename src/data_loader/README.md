# Data Loader Modules

## Overview

This directory contains the data loading and preprocessing modules for both **Named Entity Recognition (NER)** and **Relation Extraction (RE)** tasks. These modules are responsible for transforming raw `.jsonl` data into tokenized and formatted tensors suitable for training and evaluation with Hugging Face models.

Each module consists of two primary classes:
1.  A `Dataset` class (`NERDataset`, `REDataset`) that handles the core logic of reading, parsing, and transforming individual data records.
2.  A `DataModule` class (`NERDataModule`, `REDataModule`) that orchestrates the data loading process, including tokenizer initialization, dataset setup, and providing `DataLoader` instances.

---

## `ner_datamodule.py`

### Purpose

This module is designed for **token-level classification** tasks. It reads text and a list of character-level entity annotations and produces tokenized inputs where each token is assigned a label (e.g., `B-FIND`, `I-FIND`, `O`).

### Key Features

-   **BIO Labeling**: Automatically converts character-based entity spans into the standard BIO (Beginning, Inside, Outside) tagging scheme.
-   **Subword Alignment**: Aligns character-based entity spans with their corresponding tokens using the tokenizer's offset mapping. This correctly labels all subwords belonging to an entity and assigns special tokens (like `[CLS]` and `[PAD]`) a label of `-100` to be ignored during loss calculation.
-   **Dynamic Label Mapping**: Constructs a label-to-ID map from the list of entity types provided in the configuration, ensuring flexibility.
-   **Robust Validation**: Includes checks to handle records with missing or malformed entities and warns the user about entity labels present in the data but not specified in the configuration.

---

## `re_datamodule.py`

### Purpose

This module is tailored for **sequence-level classification** to identify relationships between pairs of entities. It transforms each record into multiple instances, where each instance represents a single, directional pair of entities from the original text.

### Key Features

-   **Entity Pair Generation**: Creates all possible ordered pairs of entities within a single text record (`itertools.permutations`), generating a distinct training instance for each pair.
-   **Special Token Markers**: Inserts special tokens (e.g., `[E1_START]`, `[E1_END]`, `[E2_START]`, `[E2_END]`) around the head and tail entities of a pair to provide strong positional signals to the model.
-   **"No\_Relation" Handling**: Assigns a "No\_Relation" label to entity pairs that are not explicitly related in the dataset, allowing the model to learn to distinguish between related and unrelated pairs.
-   **Configuration-Driven Filtering**: Ignores relation types found in the data but not defined in the configuration `relation_labels`, issuing a warning to the user.