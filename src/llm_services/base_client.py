from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseLLMClient(ABC):
    """
    An abstract base class that defines the standard interface for LLM clients.

    This class ensures that any LLM client, whether for OpenAI, AWS, or another
    provider, adheres to a consistent contract for making predictions. This allows
    the main application logic to remain decoupled from the specific implementation 
    details of any one provider.
    """

    @abstractmethod
    def get_ner_prediction(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Sends a prompt to the LLM and returns the extracted entities for an NER task.

        Args:
            prompt (str): The fully constructed prompt, including instructions,
                          few-shot examples, and the new text to be analyzed.

        Returns:
            List[Dict[str, Any]]: A list of extracted entity dictionaries.
                                  Each dictionary is expected to have keys like
                                  'text' and 'label'.
                                  Returns an empty list if no entities are found
                                  or in case of an error.
        """
        pass