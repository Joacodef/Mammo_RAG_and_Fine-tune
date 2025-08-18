import yaml
from typing import Dict, Any, Optional

# Add the project root to the Python path to allow for absolute imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm_services.base_client import BaseLLMClient
from src.llm_services.openai_client import OpenAIClient
from src.utils.cost_tracker import CostTracker

def get_llm_client(config_path: str = "configs/rag_config.yaml", cost_tracker: Optional[CostTracker] = None) -> BaseLLMClient:
    """
    Factory function to get an instance of an LLM client based on the provider
    specified in the configuration file.

    This function reads the configuration, determines the desired provider
    (e.g., 'openai'), and instantiates the corresponding client with its
    specific settings, including the cost tracker.

    Args:
        config_path (str): The path to the RAG configuration YAML file.
        cost_tracker (Optional[CostTracker]): An instance of the cost tracker
                                              to be passed to the client.

    Returns:
        BaseLLMClient: An instance of a class that inherits from BaseLLMClient
                       (e.g., OpenAIClient).

    Raises:
        ValueError: If the configuration file is not found or if the specified
                    provider is not supported.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found at: {config_path}")

    provider = config.get("llm", {}).get("provider")
    if not provider:
        raise ValueError("LLM provider not specified in the configuration file.")

    print(f"Initializing LLM client for provider: {provider}")

    if provider == "openai":
        provider_config = config.get("llm", {}).get("openai", {})
        return OpenAIClient(config=provider_config, cost_tracker=cost_tracker)
    
    # Placeholder for future clients like AWS SageMaker
    # elif provider == "aws_sagemaker":
    #     provider_config = config.get("llm", {}).get("aws_sagemaker", {})
    #     # from .aws_sagemaker_client import AWSSageMakerClient
    #     # return AWSSageMakerClient(config=provider_config, cost_tracker=cost_tracker)
    
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Supported providers are ['openai'].")