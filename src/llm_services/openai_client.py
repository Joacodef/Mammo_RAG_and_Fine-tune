import os
import json
import openai
from typing import List, Dict, Any, Optional

# Add the project root to the Python path to allow for absolute imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm_services.base_client import BaseLLMClient

class OpenAIClient(BaseLLMClient):
    """
    A concrete implementation of the BaseLLMClient for interacting with
    the OpenAI API (e.g., GPT-4, GPT-3.5).
    """

    def __init__(self, config: Dict[str, Any], api_key: str = None):
        """
        Initializes the OpenAI client.

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration
                                     for the OpenAI provider, including 'model'
                                     and 'temperature'.
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

    def get_ner_prediction(self, prompt: str, trace: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Sends a prompt to the OpenAI API and returns the extracted entities.

        If a trace object is provided, this method will log the API call
        as a generation nested within that trace.

        Args:
            prompt (str): The fully constructed prompt for the NER task.
            trace (Optional[Any]): The Langfuse trace object.

        Returns:
            List[Dict[str, Any]]: A list of extracted entity dictionaries.
                                  Returns an empty list in case of an API error
                                  or if the response is not valid JSON.
        """
        generation = None
        response_content = ""
        try:
            if trace:
                with trace.start_as_current_generation(
                    name="NER Prediction",
                    model=self.model,
                    input=prompt,
                    metadata={"temperature": self.temperature}
                ) as generation:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant designed to return JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"}
                    )
                    response_content = response.choices[0].message.content
                    generation.update(output=response_content, usage=response.usage)
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant designed to return JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                response_content = response.choices[0].message.content

            response_dict = json.loads(response_content)
            
            for value in response_dict.values():
                if isinstance(value, list):
                    return value
            
            return []

        except openai.APIError as e:
            error_message = f"OpenAI API returned an error: {e}"
            print(f"Error: {error_message}")
            if generation:
                generation.update(level='ERROR', status_message=error_message)
            return []
        except json.JSONDecodeError:
            error_message = f"Failed to decode JSON from model response: {response_content}"
            print(f"Error: {error_message}")
            if generation:
                generation.update(level='ERROR', status_message=error_message)
            return []
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            print(f"Error: {error_message}")
            if generation:
                generation.update(level='ERROR', status_message=error_message)
            return []