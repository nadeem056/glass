"""
Configuration and utility functions for the Ollama API wrapper.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class OllamaConfig:
    """Configuration class for Ollama client."""
    
    base_url: str = "http://localhost:11434"
    timeout: int = 30
    stream_timeout: int = 60
    retry_attempts: int = 3
    retry_backoff_factor: float = 0.3
    default_model: Optional[str] = None
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OllamaConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "stream_timeout": self.stream_timeout,
            "retry_attempts": self.retry_attempts,
            "retry_backoff_factor": self.retry_backoff_factor,
            "default_model": self.default_model,
            "default_parameters": self.default_parameters.copy(),
        }


def aggregate_streaming_response(chunks) -> str:
    """
    Utility function to aggregate streaming response chunks into complete text.
    
    Args:
        chunks: Iterator of response chunks from streaming API
        
    Returns:
        Complete aggregated text
    """
    aggregated_text = ""
    for chunk in chunks:
        if "response" in chunk:
            aggregated_text += chunk["response"]
    return aggregated_text


def format_model_info(models_response: Dict[str, Any]) -> str:
    """
    Format model information for display.
    
    Args:
        models_response: Response from list_models() API call
        
    Returns:
        Formatted string with model information
    """
    if "models" not in models_response:
        return "No models found"
    
    models = models_response["models"]
    if not models:
        return "No models available"
    
    formatted = "Available Models:\n"
    for model in models:
        name = model.get("name", "Unknown")
        size = model.get("size", 0)
        size_mb = size / (1024 * 1024) if size else 0
        formatted += f"  - {name} ({size_mb:.1f} MB)\n"
    
    return formatted.strip()


def calculate_tokens_per_second(eval_count: Optional[int], eval_duration: Optional[int]) -> Optional[float]:
    """
    Calculate tokens per second from evaluation metrics.
    
    Args:
        eval_count: Number of tokens evaluated
        eval_duration: Evaluation duration in nanoseconds
        
    Returns:
        Tokens per second, or None if calculation not possible
    """
    if eval_count and eval_duration and eval_duration > 0:
        return eval_count / (eval_duration / 1_000_000_000)
    return None


def validate_model_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize model parameters.
    
    Args:
        parameters: Raw model parameters
        
    Returns:
        Validated parameters
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not isinstance(parameters, dict):
        raise ValueError("Parameters must be a dictionary")
    
    validated = {}
    
    # Temperature validation
    if "temperature" in parameters:
        temp = parameters["temperature"]
        if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
            raise ValueError("Temperature must be a number between 0 and 2")
        validated["temperature"] = float(temp)
    
    # Top-p validation
    if "top_p" in parameters:
        top_p = parameters["top_p"]
        if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
            raise ValueError("top_p must be a number between 0 and 1")
        validated["top_p"] = float(top_p)
    
    # Top-k validation
    if "top_k" in parameters:
        top_k = parameters["top_k"]
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        validated["top_k"] = int(top_k)
    
    # Copy other parameters as-is (basic validation)
    for key, value in parameters.items():
        if key not in validated:
            if isinstance(value, (str, int, float, bool, list)):
                validated[key] = value
            else:
                raise ValueError(f"Invalid parameter type for {key}: {type(value)}")
    
    return validated