"""
Ollama API Python Wrapper

This module provides a Python wrapper for the Ollama API with support for both
streaming and non-streaming responses, proper error handling, and configurable
timeout settings.
"""

__version__ = "1.0.0"
__author__ = "Glass Project"
__email__ = "nadeem056@users.noreply.github.com"

from .client import OllamaClient
from .exceptions import OllamaError, OllamaTimeoutError, OllamaConnectionError
from .types import OllamaResponse, StreamingResponse
from .config import OllamaConfig, aggregate_streaming_response, format_model_info

__all__ = [
    "OllamaClient",
    "OllamaError",
    "OllamaTimeoutError", 
    "OllamaConnectionError",
    "OllamaResponse",
    "StreamingResponse",
    "OllamaConfig",
    "aggregate_streaming_response",
    "format_model_info",
]