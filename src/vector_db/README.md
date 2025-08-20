# Vector Database Module

## Overview

This directory contains the components for building and managing the vector database, which is the core of the retrieval mechanism in the Retrieval-Augmented Generation (RAG) pipeline. The process involves two main steps: first, converting text documents into numerical vectors (embeddings), and second, storing these embeddings in an efficient, searchable index.

---

## `sentence_embedder.py`

### Purpose

The `SentenceEmbedder` class serves as a straightforward wrapper for a `sentence-transformers` model from the Hugging Face Hub. Its sole responsibility is to load a specified pre-trained model and provide a clean interface for converting text into high-dimensional vector embeddings.

### Key Features

-   **Model Loading**: Initializes a `SentenceTransformer` model with a given model name (e.g., `'all-MiniLM-L6-v2'`).
-   **Device Management**: Automatically detects and utilizes a CUDA-enabled GPU if one is available, otherwise falling back to the CPU for computations.
-   **Embedding Generation**: Provides a simple `embed` method that takes a list of texts and returns a NumPy array where each row is the vector embedding for the corresponding text.

---

## `database_manager.py`

### Purpose

The `DatabaseManager` class orchestrates the entire lifecycle of the **FAISS** vector database. It uses the `SentenceEmbedder` to convert source documents into vectors and then builds a searchable index from these vectors. This index is used by the RAG pipeline to find relevant examples for constructing few-shot prompts.

### Key Features

-   **Index Lifecycle Management**: Handles the creation, saving, and loading of a FAISS index file. The `build_index` method will intelligently load an existing index or create a new one if it doesn't exist or if a rebuild is forced.
-   **Data Ingestion**: Reads a source `.jsonl` data file and uses the provided `SentenceEmbedder` instance to generate embeddings for all the text records.
-   **Similarity Search**: Implements a `search` method that takes a new query text, embeds it, and performs a similarity search against the FAISS index to retrieve the `top_k` most similar documents from the original source data.