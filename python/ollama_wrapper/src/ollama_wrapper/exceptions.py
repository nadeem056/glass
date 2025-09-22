"""
Custom exceptions for the Ollama API wrapper.
"""

from typing import Optional


class OllamaError(Exception):
    """Base exception for Ollama API wrapper."""
    
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class OllamaTimeoutError(OllamaError):
    """Raised when an API request times out."""
    
    def __init__(self, message: str = "Request timed out") -> None:
        super().__init__(message)


class OllamaConnectionError(OllamaError):
    """Raised when there's a connection error to the Ollama API."""
    
    def __init__(self, message: str = "Connection error") -> None:
        super().__init__(message)


class OllamaInvalidResponseError(OllamaError):
    """Raised when the API returns an invalid response."""
    
    def __init__(self, message: str = "Invalid response from API") -> None:
        super().__init__(message)