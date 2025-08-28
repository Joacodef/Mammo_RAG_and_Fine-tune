# LLM Services Module

## Overview

This directory contains the components responsible for interacting with external Large Language Model (LLM) providers. The module is designed with a **factory pattern** to be modular and provider-agnostic. This allows the main application to easily switch between different LLMs (e.g., from OpenAI to AWS SageMaker) by simply changing a configuration setting, without altering the core application code.

---

## `__init__.py`

### Purpose

This file contains the `get_llm_client` factory function, which serves as the single entry point for the rest of the application to obtain an LLM client.

### Key Features

-   **Configuration-Driven**: The factory reads the `rag_config.yaml` file to determine which LLM provider is specified (e.g., `provider: "openai"`).
-   **Dynamic Instantiation**: Based on the configuration, it dynamically instantiates and returns the corresponding concrete client (e.g., `OpenAIClient`). This decouples the client-calling code from the client implementation.

---

## `base_client.py`

### Purpose

This file defines the `BaseLLMClient` abstract base class. It serves as a strict contract or interface that all concrete LLM clients must adhere to.

### Key Features

-   **Standardized Interface**: It declares the abstract method `get_ner_prediction`, ensuring that every client implementation will have a consistent method for performing NER tasks. This method is designed to accept an optional `trace` object to support observability platforms like Langfuse.

---

## `openai_client.py`

### Purpose

This file provides the concrete implementation, `OpenAIClient`, for interacting with the OpenAI API.

### Key Features

-   **API Interaction**: Handles all the logic for sending requests to OpenAI's chat completions endpoint, including formatting the messages and specifying the JSON response format.
-   **Authentication**: Manages the API key, retrieving it either from an environment variable (`OPENAI_API_KEY`) or from a direct argument.
-   **Observability and Tracing**: Integrates with Langfuse for detailed tracing of LLM calls. The `get_ner_prediction` method can receive a trace object to log the prompt, final response, token usage, and any potential errors as a distinct generation step.
-   **Robust Error Handling**: Includes `try-except` blocks to gracefully handle potential issues like API errors or JSON decoding failures, preventing them from crashing the prediction pipeline.