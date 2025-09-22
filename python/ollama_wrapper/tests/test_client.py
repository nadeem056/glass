"""
Tests for the Ollama API wrapper.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import ConnectionError, Timeout

# Add the src directory to Python path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from ollama_wrapper import (
    OllamaClient, 
    OllamaError, 
    OllamaTimeoutError, 
    OllamaConnectionError,
    OllamaResponse
)


class TestOllamaClient:
    """Test cases for OllamaClient."""
    
    def test_init_default_params(self):
        """Test client initialization with default parameters."""
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.timeout == 30
        assert client.stream_timeout == 60
    
    def test_init_custom_params(self):
        """Test client initialization with custom parameters."""
        client = OllamaClient(
            base_url="http://custom:8080",
            timeout=60,
            stream_timeout=120
        )
        assert client.base_url == "http://custom:8080"
        assert client.timeout == 60
        assert client.stream_timeout == 120
    
    def test_make_request_url(self):
        """Test URL construction."""
        client = OllamaClient(base_url="http://localhost:11434")
        
        # Test with leading slash
        url = client._make_request_url("/api/generate")
        assert url == "http://localhost:11434/api/generate"
        
        # Test without leading slash
        url = client._make_request_url("api/generate")
        assert url == "http://localhost:11434/api/generate"
    
    def test_parse_streaming_line_valid(self):
        """Test parsing valid streaming response line."""
        client = OllamaClient()
        
        line = '{"response": "Hello", "done": false}'
        result = client._parse_streaming_line(line)
        
        assert result == {"response": "Hello", "done": False}
    
    def test_parse_streaming_line_empty(self):
        """Test parsing empty streaming response line."""
        client = OllamaClient()
        
        result = client._parse_streaming_line("")
        assert result is None
        
        result = client._parse_streaming_line("   ")
        assert result is None
    
    def test_parse_streaming_line_invalid_json(self):
        """Test parsing invalid JSON in streaming response."""
        client = OllamaClient()
        
        with pytest.raises(Exception):  # Should raise OllamaInvalidResponseError
            client._parse_streaming_line("invalid json")
    
    @patch('ollama_wrapper.client.requests.Session.post')
    def test_generate_streaming_success(self, mock_post):
        """Test successful streaming generation."""
        # Mock streaming response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.iter_lines.return_value = [
            '{"response": "Hello", "done": false}',
            '{"response": " world", "done": false}',
            '{"response": "!", "done": true, "model": "test"}'
        ]
        
        mock_post.return_value.__enter__.return_value = mock_response
        
        client = OllamaClient()
        
        chunks = list(client.generate_streaming("test-model", "Hello"))
        
        assert len(chunks) == 3
        assert chunks[0]["response"] == "Hello"
        assert chunks[1]["response"] == " world"
        assert chunks[2]["response"] == "!"
        assert chunks[2]["done"] is True
    
    @patch('ollama_wrapper.client.requests.Session.post')
    def test_generate_streaming_error_response(self, mock_post):
        """Test streaming generation with error response."""
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 404
        mock_response.text = "Model not found"
        
        mock_post.return_value.__enter__.return_value = mock_response
        
        client = OllamaClient()
        
        with pytest.raises(OllamaError) as exc_info:
            list(client.generate_streaming("nonexistent-model", "Hello"))
        
        assert "404" in str(exc_info.value)
        assert exc_info.value.status_code == 404
    
    @patch('ollama_wrapper.client.requests.Session.post')
    def test_generate_streaming_timeout(self, mock_post):
        """Test streaming generation timeout."""
        mock_post.side_effect = Timeout()
        
        client = OllamaClient()
        
        with pytest.raises(OllamaTimeoutError):
            list(client.generate_streaming("test-model", "Hello"))
    
    @patch('ollama_wrapper.client.requests.Session.post')
    def test_generate_streaming_connection_error(self, mock_post):
        """Test streaming generation connection error."""
        mock_post.side_effect = ConnectionError()
        
        client = OllamaClient()
        
        with pytest.raises(OllamaConnectionError):
            list(client.generate_streaming("test-model", "Hello"))
    
    def test_generate_non_streaming_success(self):
        """Test successful non-streaming generation."""
        client = OllamaClient()
        
        # Mock the streaming method to return test data
        def mock_streaming(model, prompt, parameters=None):
            yield {"response": "Hello", "done": False}
            yield {"response": " world", "done": False}
            yield {
                "response": "!", 
                "done": True, 
                "model": "test-model",
                "total_duration": 1000000000,
                "eval_count": 10
            }
        
        client.generate_streaming = mock_streaming
        
        response = client.generate_non_streaming("test-model", "Hello")
        
        assert isinstance(response, OllamaResponse)
        assert response.text == "Hello world!"
        assert response.model == "test-model"
        assert response.done is True
        assert response.eval_count == 10
    
    def test_generate_non_streaming_no_response(self):
        """Test non-streaming generation with no response."""
        client = OllamaClient()
        
        # Mock empty streaming response
        client.generate_streaming = lambda *args, **kwargs: iter([])
        
        with pytest.raises(Exception):  # Should raise OllamaInvalidResponseError
            client.generate_non_streaming("test-model", "Hello")
    
    def test_generate_convenience_streaming(self):
        """Test convenience generate method in streaming mode."""
        client = OllamaClient()
        
        def mock_streaming(model, prompt, parameters=None):
            yield {"response": "test", "done": True}
        
        client.generate_streaming = mock_streaming
        
        result = client.generate("test-model", "Hello", stream=True)
        chunks = list(result)
        
        assert len(chunks) == 1
        assert chunks[0]["response"] == "test"
    
    def test_generate_convenience_non_streaming(self):
        """Test convenience generate method in non-streaming mode."""
        client = OllamaClient()
        
        mock_response = OllamaResponse(
            text="Hello world", 
            model="test-model", 
            done=True
        )
        client.generate_non_streaming = lambda *args, **kwargs: mock_response
        
        result = client.generate("test-model", "Hello", stream=False)
        
        assert isinstance(result, OllamaResponse)
        assert result.text == "Hello world"
    
    @patch('ollama_wrapper.client.requests.Session.get')
    def test_list_models_success(self, mock_get):
        """Test successful model listing."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2", "size": 1000000},
                {"name": "codellama", "size": 2000000}
            ]
        }
        mock_get.return_value = mock_response
        
        client = OllamaClient()
        result = client.list_models()
        
        assert "models" in result
        assert len(result["models"]) == 2
        assert result["models"][0]["name"] == "llama2"
    
    @patch('ollama_wrapper.client.requests.Session.get')
    def test_list_models_error(self, mock_get):
        """Test model listing error."""
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_get.return_value = mock_response
        
        client = OllamaClient()
        
        with pytest.raises(OllamaError) as exc_info:
            client.list_models()
        
        assert "500" in str(exc_info.value)
        assert exc_info.value.status_code == 500
    
    @patch('ollama_wrapper.client.requests.Session.get')
    def test_health_check_healthy(self, mock_get):
        """Test health check when API is healthy."""
        mock_response = Mock()
        mock_response.ok = True
        mock_get.return_value = mock_response
        
        client = OllamaClient()
        result = client.health_check()
        
        assert result is True
    
    @patch('ollama_wrapper.client.requests.Session.get')
    def test_health_check_unhealthy(self, mock_get):
        """Test health check when API is unhealthy."""
        mock_response = Mock()
        mock_response.ok = False
        mock_get.return_value = mock_response
        
        client = OllamaClient()
        result = client.health_check()
        
        assert result is False
    
    @patch('ollama_wrapper.client.requests.Session.get')
    def test_health_check_connection_error(self, mock_get):
        """Test health check with connection error."""
        mock_get.side_effect = ConnectionError()
        
        client = OllamaClient()
        result = client.health_check()
        
        assert result is False
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with OllamaClient() as client:
            assert isinstance(client, OllamaClient)
            # Mock the close method to verify it's called
            client.close = Mock()
        
        client.close.assert_called_once()


class TestOllamaResponse:
    """Test cases for OllamaResponse."""
    
    def test_from_dict_complete(self):
        """Test creating OllamaResponse from complete dictionary."""
        data = {
            "response": "Hello world",
            "model": "llama2",
            "done": True,
            "context": [1, 2, 3],
            "total_duration": 1000000000,
            "load_duration": 500000000,
            "eval_count": 10,
            "eval_duration": 200000000
        }
        
        response = OllamaResponse.from_dict(data)
        
        assert response.text == "Hello world"
        assert response.model == "llama2"
        assert response.done is True
        assert response.context == [1, 2, 3]
        assert response.total_duration == 1000000000
        assert response.eval_count == 10
    
    def test_from_dict_minimal(self):
        """Test creating OllamaResponse from minimal dictionary."""
        data = {
            "response": "Hi",
            "model": "test",
            "done": False
        }
        
        response = OllamaResponse.from_dict(data)
        
        assert response.text == "Hi"
        assert response.model == "test"
        assert response.done is False
        assert response.context is None
        assert response.total_duration is None
    
    def test_from_dict_empty(self):
        """Test creating OllamaResponse from empty dictionary."""
        data = {}
        
        response = OllamaResponse.from_dict(data)
        
        assert response.text == ""
        assert response.model == ""
        assert response.done is False
        assert response.context is None