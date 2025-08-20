import pytest
import json
from unittest.mock import patch, MagicMock
import openai

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.llm_services.openai_client import OpenAIClient
from src.utils.cost_tracker import CostTracker

# --- Fixtures for Testing OpenAIClient ---

@pytest.fixture
def base_config():
    """Provides a base configuration for the client."""
    return {"model": "test-model", "temperature": 0.5}

@pytest.fixture
def mock_cost_tracker():
    """Provides a mock CostTracker instance."""
    return MagicMock(spec=CostTracker)

@pytest.fixture
def mock_success_response():
    """Creates a mock successful API response object."""
    response = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    
    response_content = json.dumps({"entities": [{"text": "finding", "label": "FIND"}]})
    response.choices = [MagicMock(message=MagicMock(content=response_content))]
    
    return response

@patch('src.llm_services.openai_client.openai.OpenAI')
def test_initialization_with_api_key(mock_openai_class, base_config):
    """
    Tests that the client initializes correctly when an API key is provided directly.
    """
    client = OpenAIClient(config=base_config, api_key="test_key")
    mock_openai_class.assert_called_once_with(api_key="test_key")
    assert client.model == "test-model"

@patch('src.llm_services.openai_client.openai.OpenAI')
def test_initialization_with_env_var(mock_openai_class, base_config, monkeypatch):
    """
    Tests that the client initializes correctly using an environment variable for the API key.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "env_test_key")
    client = OpenAIClient(config=base_config)
    mock_openai_class.assert_called_once_with(api_key="env_test_key")

def test_initialization_raises_error_without_key(base_config, monkeypatch):
    """
    Tests that a ValueError is raised if no API key is provided via argument or environment variable.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OpenAI API key must be provided"):
        OpenAIClient(config=base_config)

@patch('src.llm_services.openai_client.openai.OpenAI')
def test_get_ner_prediction_success(mock_openai_class, base_config, mock_success_response):
    """
    Tests the happy path for get_ner_prediction, ensuring it parses and returns entities.
    """
    mock_client_instance = mock_openai_class.return_value
    mock_client_instance.chat.completions.create.return_value = mock_success_response
    
    client = OpenAIClient(config=base_config, api_key="test_key")
    result = client.get_ner_prediction("some prompt")
    
    assert result == [{"text": "finding", "label": "FIND"}]
    mock_client_instance.chat.completions.create.assert_called_once()

@patch('src.llm_services.openai_client.openai.OpenAI')
def test_get_ner_prediction_logs_cost(mock_openai_class, base_config, mock_cost_tracker, mock_success_response):
    """
    Tests that get_ner_prediction correctly logs the request with the CostTracker.
    """
    mock_client_instance = mock_openai_class.return_value
    mock_client_instance.chat.completions.create.return_value = mock_success_response

    client = OpenAIClient(config=base_config, api_key="test_key", cost_tracker=mock_cost_tracker)
    client.get_ner_prediction("some prompt")
    
    mock_cost_tracker.log_request.assert_called_once_with(
        model="test-model",
        prompt_tokens=100,
        completion_tokens=50
    )

@patch('src.llm_services.openai_client.openai.OpenAI')
def test_get_ner_prediction_handles_api_error(mock_openai_class, base_config):
    """
    Tests that get_ner_prediction returns an empty list when an openai.APIError occurs.
    """
    mock_client_instance = mock_openai_class.return_value
    # The constructor for APIError in recent versions requires a 'request' object.
    mock_request = MagicMock()
    mock_client_instance.chat.completions.create.side_effect = openai.APIError(
        "API Error", request=mock_request, body=None
    )

    client = OpenAIClient(config=base_config, api_key="test_key")
    result = client.get_ner_prediction("some prompt")
    
    assert result == []

@patch('src.llm_services.openai_client.openai.OpenAI')
def test_get_ner_prediction_handles_json_decode_error(mock_openai_class, base_config):
    """
    Tests that get_ner_prediction returns an empty list if the API response is not valid JSON.
    """
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="this is not json"))]
    mock_response.usage = None # No usage info if parsing fails before that

    mock_client_instance = mock_openai_class.return_value
    mock_client_instance.chat.completions.create.return_value = mock_response
    
    client = OpenAIClient(config=base_config, api_key="test_key")
    result = client.get_ner_prediction("some prompt")
    
    assert result == []

@patch('src.llm_services.openai_client.openai.OpenAI')
def test_get_ner_prediction_handles_unexpected_structure(mock_openai_class, base_config):
    """
    Tests that get_ner_prediction returns an empty list if the JSON does not contain a list value.
    """
    mock_response = MagicMock()
    response_content = json.dumps({"key": "value", "another_key": 123})
    mock_response.choices = [MagicMock(message=MagicMock(content=response_content))]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    mock_client_instance = mock_openai_class.return_value
    mock_client_instance.chat.completions.create.return_value = mock_response

    client = OpenAIClient(config=base_config, api_key="test_key")
    result = client.get_ner_prediction("some prompt")

    assert result == []