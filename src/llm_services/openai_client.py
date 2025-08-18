import os
import json
import openai
from typing import List, Dict, Any, Optional

# Add the project root to the Python path to allow for absolute imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm_services.base_client import BaseLLMClient
from src.utils.cost_tracker import CostTracker

class OpenAIClient(BaseLLMClient):
    """
    A concrete implementation of the BaseLLMClient for interacting with
    the OpenAI API (e.g., GPT-4, GPT-3.5).
    """

    def __init__(self, config: Dict[str, Any], cost_tracker: Optional[CostTracker] = None, api_key: str = None):
        """
        Initializes the OpenAI client.

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration
                                     for the OpenAI provider, including 'model'
                                     and 'temperature'.
            cost_tracker (Optional[CostTracker]): An instance of the CostTracker
                                                  to log API usage and costs.
            api_key (str, optional): The OpenAI API key. If not provided, it will
                                     be read from the OPENAI_API_KEY environment
                                     variable.
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided or set as an environment variable (OPENAI_API_KEY).")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = config.get("model", "gpt-4o")
        self.temperature = config.get("temperature", 0.1)
        self.cost_tracker = cost_tracker

    def get_ner_prediction(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Sends a prompt to the OpenAI API and returns the extracted entities.

        If a cost_tracker is configured, this method will also log the token
        usage and estimated cost of the API call.

        Args:
            prompt (str): The fully constructed prompt for the NER task.

        Returns:
            List[Dict[str, Any]]: A list of extracted entity dictionaries.
                                  Returns an empty list in case of an API error
                                  or if the response is not valid JSON.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant designed to return JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            # --- Log the request with the CostTracker ---
            if self.cost_tracker and response.usage:
                self.cost_tracker.log_request(
                    model=self.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens
                )
            
            response_content = response.choices[0].message.content
            response_dict = json.loads(response_content)
            
            for value in response_dict.values():
                if isinstance(value, list):
                    return value
            
            return []

        except openai.APIError as e:
            print(f"Error: OpenAI API returned an error: {e}")
            return []
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from model response: {response.choices[0].message.content}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return []