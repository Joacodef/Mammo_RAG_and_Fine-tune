# src/llm_services/ollama_client.py
import json
import re
import urllib.request
from typing import List, Dict, Any, Optional

# Add the project root to the Python path to allow for absolute imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm_services.base_client import BaseLLMClient

def safe_print(msg: str):
    encoding = sys.stdout.encoding or 'utf-8'
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            print(msg.encode(encoding, errors='replace').decode(encoding))
        except Exception:
            print(msg.encode('ascii', errors='replace').decode('ascii'))


class OllamaClient(BaseLLMClient):
    """
    A concrete implementation of the BaseLLMClient for interacting with a local
    or tundeled Ollama server via direct REST API calls.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Ollama client.

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration
                                     for the Ollama provider, including 'model',
                                     'base_url', and 'llm_parameters'.
        """
        self.model_name = config.get("model", "llama3:8b")
        self.base_url = config.get("base_url", "http://localhost:11434").rstrip('/')
        self.llm_params = config.get("llm_parameters", {})
        print(f"Ollama client initialized for model '{self.model_name}' at {self.base_url} (Direct REST mode)")

    def _call_api(self, prompt: str) -> str:
        """Sends a POST request to Ollama generate endpoint."""
        url = f"{self.base_url}/api/generate"
        
        # Merge config parameters with options
        options = {
            "temperature": 0.1,
            "stop": ["\n[", "]\n[", "]\n\n", "\n---"]
        }
        if self.llm_params:
            options.update(self.llm_params)
            
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": options
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            return res_data.get('response', '')

    def _get_prediction(self, prompt: str, trace: Optional[Any] = None, report_index: int = -1, task: str = "NER") -> List[Dict[str, Any]]:
        """
        A helper method to send a prompt to the Ollama model and parse the JSON response.
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
                    response_content = self._call_api(prompt)
                    # Log the raw, potentially messy, model output first
                    generation.update(output={"raw_output": response_content})
            else:
                response_content = self._call_api(prompt)

            # Use regex to find a JSON list or object within the response text
            # This handles cases where the model adds conversational text around the JSON
            json_match = re.search(r'\[.*?\]|\{.*?\}', response_content, re.DOTALL)
            
            if not json_match:
                raise json.JSONDecodeError("No JSON array or object found in the model's response.", response_content, 0)
            
            json_string = json_match.group(0)
            parsed_response = json.loads(json_string)

            # Update the generation with the clean, parsed output for better observability
            if generation:
                generation.update(output=parsed_response)
            
            # If the model returns a JSON list directly, return it.
            if isinstance(parsed_response, list):
                return parsed_response

            # If the model returns a JSON object containing a list, find and return the list.
            if isinstance(parsed_response, dict):
                for value in parsed_response.values():
                    if isinstance(value, list):
                        return value
            
            # If the format is unexpected, log a warning and return empty.
            safe_print(f"Warning: Model returned valid JSON, but it was not a list or a dict containing a list. Response: {json_string}")
            return []

        except json.JSONDecodeError:
            error_message = f"Failed to decode JSON from Ollama model response. Here is the raw output for the given text:\n {response_content}"
            safe_print(f"Error: {error_message}")
            if generation:
                generation.update(level='ERROR', status_message=error_message)
            return []
        except Exception as e:
            error_message = f"An unexpected error occurred while calling Ollama: {e}"
            safe_print(f"Error: {error_message}")
            if generation:
                generation.update(level='ERROR', status_message=error_message)
            return []

    def get_ner_prediction(self, prompt: str, trace: Optional[Any] = None, report_index: int = -1) -> List[Dict[str, Any]]:
        """
        Sends a prompt to the Ollama model and returns the extracted entities for an NER task.
        """
        return self._get_prediction(prompt, trace, report_index, task="NER")

    def get_re_prediction(self, prompt: str, trace: Optional[Any] = None, report_index: int = -1) -> List[Dict[str, Any]]:
        """
        Sends a prompt to the Ollama model and returns the extracted relations for an RE task.
        """
        return self._get_prediction(prompt, trace, report_index, task="RE")