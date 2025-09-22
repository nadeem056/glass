"""
Type definitions for the Ollama API wrapper.
"""

from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional, Union
import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias


@dataclass
class OllamaResponse:
    """Standard response from Ollama API."""
    
    text: str
    model: str
    done: bool
    context: Optional[list] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OllamaResponse":
        """Create OllamaResponse from API response dictionary."""
        return cls(
            text=data.get("response", ""),
            model=data.get("model", ""),
            done=data.get("done", False),
            context=data.get("context"),
            total_duration=data.get("total_duration"),
            load_duration=data.get("load_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            prompt_eval_duration=data.get("prompt_eval_duration"),
            eval_count=data.get("eval_count"),
            eval_duration=data.get("eval_duration"),
        )


# Type alias for streaming response generator
StreamingResponse: TypeAlias = Generator[Dict[str, Any], None, None]

# Type alias for model parameters
ModelParameters: TypeAlias = Dict[str, Union[str, int, float, bool]]