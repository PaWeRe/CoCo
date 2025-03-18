from fastapi.testclient import TestClient
import pytest
import json
from unittest.mock import patch
import os
import sys
from pathlib import Path

# Add the root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Set a dummy API key before importing the app (for easier testing)
os.environ["OPENAI_API_KEY"] = "dummy-key"

# Now import the app
from backend import app

client = TestClient(app)


# Add this fixture at the top of your test file
@pytest.fixture(autouse=True)
def mock_openai():
    with patch("backend.openai_client") as mock:
        # Mock the models.list method
        class MockModel:
            def __init__(self, id):
                self.id = id
                self.object = "model"

        class MockModelList:
            def __init__(self):
                self.data = [MockModel("test-model")]
                self.object = "list"

        # Set the return value for models.list
        mock.models.list.return_value = MockModelList()

        # Mock the chat.completions.create method
        class MockMessage:
            def __init__(self):
                self.role = "assistant"
                self.content = "test response"

        class MockChoice:
            def __init__(self):
                self.message = MockMessage()
                self.finish_reason = "stop"
                self.index = 0

        class MockUsage:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20
                self.total_tokens = 30

        class MockCompletion:
            def __init__(self):
                self.id = "chatcmpl-test"
                self.object = "chat.completion"
                self.created = 1234567890
                self.model = "test-model"
                self.choices = [MockChoice()]
                self.usage = MockUsage()

        # Set the return value for chat.completions.create
        mock.chat.completions.create.return_value = MockCompletion()

        yield mock


def test_models_endpoint():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0
    assert data["data"][0]["id"] == "test-model"


def test_intent_discover_endpoint():
    payload = {"messages": [{"role": "user", "content": "I want to code."}]}
    response = client.post("/v1/intent/discover", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Check that the intent is detected as code assistance
    assert "intent" in data
    assert data["intent"] == "code_assistance"


def test_context_retrieve_endpoint():
    payload = {"intent": "code_assistance"}
    response = client.post("/v1/context/retrieve", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Check that the context is included in the response
    assert "context" in data


def test_preferences_retrieve_endpoint():
    payload = {"intent": "code_assistance"}
    response = client.post("/v1/preferences/retrieve", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Verify that the preferences are present
    assert "preferences" in data


def test_chat_completions_non_stream():
    payload = {
        "model": "text-davinci-003",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50,
        "temperature": 0.5,
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert data["choices"][0]["message"]["content"] == "test response"
