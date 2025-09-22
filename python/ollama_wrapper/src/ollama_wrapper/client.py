"""
Ollama API client implementation with streaming and non-streaming support.
"""

import json
import time
from typing import Any, Dict, Generator, Optional, Union
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from .exceptions import (
    OllamaConnectionError,
    OllamaError,
    OllamaInvalidResponseError,
    OllamaTimeoutError,
)
from .types import ModelParameters, OllamaResponse, StreamingResponse


class OllamaClient:
    """
    Python client for Ollama API with support for streaming and non-streaming modes.
    
    This client provides methods to interact with the Ollama API, supporting both
    streaming word-by-word responses and aggregated non-streaming responses.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_backoff_factor: float = 0.3,
        stream_timeout: int = 60,
    ) -> None:
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL for the Ollama API (default: http://localhost:11434)
            timeout: Timeout for non-streaming requests in seconds (default: 30)
            retry_attempts: Number of retry attempts for failed requests (default: 3)
            retry_backoff_factor: Backoff factor for retries (default: 0.3)
            stream_timeout: Timeout for streaming requests in seconds (default: 60)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.stream_timeout = stream_timeout
        
        # Create session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=retry_attempts,
            backoff_factor=retry_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
    
    def _make_request_url(self, endpoint: str) -> str:
        """Create full request URL from endpoint."""
        return urljoin(f"{self.base_url}/", endpoint.lstrip("/"))
    
    def _handle_request_error(self, error: requests.RequestException) -> None:
        """Handle and convert request errors to appropriate Ollama exceptions."""
        if isinstance(error, requests.exceptions.Timeout):
            raise OllamaTimeoutError(f"Request timed out: {error}")
        elif isinstance(error, requests.exceptions.ConnectionError):
            raise OllamaConnectionError(f"Connection error: {error}")
        else:
            raise OllamaError(f"Request failed: {error}")
    
    def _parse_streaming_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single line from streaming response."""
        line = line.strip()
        if not line:
            return None
            
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise OllamaInvalidResponseError(f"Invalid JSON in streaming response: {e}")
    
    def generate_streaming(
        self,
        model: str,
        prompt: str,
        parameters: Optional[ModelParameters] = None,
    ) -> StreamingResponse:
        """
        Generate streaming response from Ollama API.
        
        Args:
            model: Name of the model to use
            prompt: Input prompt for the model
            parameters: Optional model parameters
            
        Returns:
            Generator yielding response chunks as dictionaries
            
        Raises:
            OllamaError: If the API request fails
            OllamaTimeoutError: If the request times out
            OllamaConnectionError: If there's a connection error
        """
        url = self._make_request_url("/api/generate")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
        }
        
        if parameters:
            payload.update(parameters)
        
        try:
            with self.session.post(
                url,
                json=payload,
                stream=True,
                timeout=self.stream_timeout,
            ) as response:
                
                if not response.ok:
                    raise OllamaError(
                        f"API request failed with status {response.status_code}: {response.text}",
                        status_code=response.status_code,
                    )
                
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        parsed_line = self._parse_streaming_line(line)
                        if parsed_line:
                            yield parsed_line
                            
        except requests.RequestException as e:
            self._handle_request_error(e)
    
    def generate_non_streaming(
        self,
        model: str,
        prompt: str,
        parameters: Optional[ModelParameters] = None,
    ) -> OllamaResponse:
        """
        Generate non-streaming response from Ollama API.
        
        This method aggregates streaming responses into a single complete response.
        
        Args:
            model: Name of the model to use
            prompt: Input prompt for the model
            parameters: Optional model parameters
            
        Returns:
            Complete aggregated response as OllamaResponse object
            
        Raises:
            OllamaError: If the API request fails
            OllamaTimeoutError: If the request times out
            OllamaConnectionError: If there's a connection error
        """
        aggregated_text = ""
        last_chunk = None
        
        for chunk in self.generate_streaming(model, prompt, parameters):
            if "response" in chunk:
                aggregated_text += chunk["response"]
            last_chunk = chunk
        
        if not last_chunk:
            raise OllamaInvalidResponseError("No response received from API")
        
        # Create final response with aggregated text
        final_response = last_chunk.copy()
        final_response["response"] = aggregated_text
        
        return OllamaResponse.from_dict(final_response)
    
    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        parameters: Optional[ModelParameters] = None,
    ) -> Union[OllamaResponse, StreamingResponse]:
        """
        Generate response from Ollama API.
        
        This is a convenience method that automatically chooses between streaming
        and non-streaming based on the stream parameter.
        
        Args:
            model: Name of the model to use
            prompt: Input prompt for the model
            stream: If True, return streaming response; if False, return aggregated response
            parameters: Optional model parameters
            
        Returns:
            Either a StreamingResponse generator or complete OllamaResponse object
            
        Raises:
            OllamaError: If the API request fails
            OllamaTimeoutError: If the request times out
            OllamaConnectionError: If there's a connection error
        """
        if stream:
            return self.generate_streaming(model, prompt, parameters)
        else:
            return self.generate_non_streaming(model, prompt, parameters)
    
    def list_models(self) -> Dict[str, Any]:
        """
        List available models.
        
        Returns:
            Dictionary containing available models
            
        Raises:
            OllamaError: If the API request fails
        """
        url = self._make_request_url("/api/tags")
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            
            if not response.ok:
                raise OllamaError(
                    f"API request failed with status {response.status_code}: {response.text}",
                    status_code=response.status_code,
                )
            
            return response.json()
            
        except requests.RequestException as e:
            self._handle_request_error(e)
    
    def health_check(self) -> bool:
        """
        Check if Ollama API is healthy.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            url = self._make_request_url("/api/tags")
            response = self.session.get(url, timeout=5)
            return response.ok
        except requests.RequestException:
            return False
    
    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()
    
    def __enter__(self) -> "OllamaClient":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()