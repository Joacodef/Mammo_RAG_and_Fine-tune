# src/llm_services/ollama_client.py
import json
from typing import List, Dict, Any, Optional

# Add the project root to the Python path to allow for absolute imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_community.llms import Ollama
from src.llm_services.base_client import BaseLLMClient

class OllamaClient(BaseLLMClient):
    """
    A concrete implementation of the BaseLLMClient for interacting with a local
    Ollama server.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Ollama client.

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration
                                     for the Ollama provider, including 'model',
                                     'base_url', and 'llm_parameters'.
        """
        self.model_name = config.get("model", "llama2")
        base_url = config.get("base_url", "http://localhost:11434")
        llm_params = config.get("llm_parameters", {})

        # These parameters are passed directly to the Ollama constructor
        self.client = Ollama(
            model=self.model_name,
            base_url=base_url,
            **llm_params
        )
        print(f"Ollama client initialized for model '{self.model_name}' at {base_url}")

    def _get_prediction(self, prompt: str, trace: Optional[Any] = None, report_index: int = -1, task: str = "NER") -> List[Dict[str, Any]]:
        """
        A helper method to send a prompt to the Ollama model and parse the JSON response.
        
        Args:
            prompt (str): The fully constructed prompt for the task.
            trace (Optional[Any]): The Langfuse trace object.
            report_index (int): The index of the report in the test set for tracing.
            task (str): The name of the task ("NER" or "RE") for tracing purposes.

        Returns:
            List[Dict[str, Any]]: A list of extracted entity or relation dictionaries.
                                  Returns an empty list in case of an error.
        """
        generation = None
        response_content = ""
        try:
            if trace:
                 with trace.start_as_current_generation(
                    name=f"{task} Prediction (Report {report_index})",
                    model=self.model_name,
                    input=prompt
                ) as generation:
                    response_content = self.client.invoke(prompt)
                    generation.update(output=response_content)
            else:
                response_content = self.client.invoke(prompt)

            parsed_response = json.loads(response_content)
            
            # If the model returns a JSON list directly, return it.
            if isinstance(parsed_response, list):
                return parsed_response

            # If the model returns a JSON object containing a list, find and return the list.
            if isinstance(parsed_response, dict):
                for value in parsed_response.values():
                    if isinstance(value, list):
                        return value
            
            # If the format is unexpected, log a warning and return empty.
            print(f"Warning: Model returned valid JSON, but it was not a list or a dict containing a list. Response: {response_content}")
            return []

        except json.JSONDecodeError:
            error_message = f"Failed to decode JSON from Ollama model response: {response_content}"
            print(f"Error: {error_message}")
            if generation:
                generation.update(level='ERROR', status_message=error_message)
            return []
        except Exception as e:
            error_message = f"An unexpected error occurred while calling Ollama: {e}"
            print(f"Error: {error_message}")
            if generation:
                generation.update(level='ERROR', status_message=error_message)
            return []