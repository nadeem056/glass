# Ollama API Python Wrapper

A Python wrapper for the Ollama API that supports both streaming and non-streaming responses, with proper error handling and configurable timeout settings.

## Features

- **Dual Mode Support**: Both streaming and non-streaming API calls
- **Response Aggregation**: Automatically aggregates streaming word-by-word responses into complete text
- **Error Handling**: Comprehensive error handling for API failures, timeouts, and connection errors
- **Configurable Timeouts**: Separate timeout configurations for streaming and non-streaming requests
- **Retry Logic**: Built-in retry mechanism for failed requests
- **Type Hints**: Full type annotation support for better IDE experience
- **Context Manager**: Support for context manager usage
- **Health Checking**: Built-in health check functionality

## Installation

### From Source

```bash
cd python/ollama_wrapper
pip install -e .
```

### Development Installation

```bash
cd python/ollama_wrapper
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from ollama_wrapper import OllamaClient

# Initialize the client
client = OllamaClient(base_url="http://localhost:11434")

# Non-streaming response (aggregated)
response = client.generate_non_streaming(
    model="llama2",
    prompt="What is the capital of France?",
    parameters={"temperature": 0.7}
)
print(response.text)

# Streaming response (word-by-word)
for chunk in client.generate_streaming(
    model="llama2", 
    prompt="Tell me a story"
):
    if "response" in chunk:
        print(chunk["response"], end="", flush=True)
    if chunk.get("done"):
        break
```

### Using the Convenience Method

```python
from ollama_wrapper import OllamaClient

with OllamaClient() as client:
    # Non-streaming
    response = client.generate(
        model="llama2",
        prompt="Hello",
        stream=False
    )
    
    # Streaming
    for chunk in client.generate(
        model="llama2",
        prompt="Hello", 
        stream=True
    ):
        # Process chunk
        pass
```

## Configuration Options

```python
client = OllamaClient(
    base_url="http://localhost:11434",      # Ollama API base URL
    timeout=30,                             # Non-streaming timeout (seconds)
    stream_timeout=60,                      # Streaming timeout (seconds)
    retry_attempts=3,                       # Number of retry attempts
    retry_backoff_factor=0.3               # Backoff factor for retries
)
```

## API Reference

### OllamaClient

The main client class for interacting with the Ollama API.

#### Methods

##### `generate_streaming(model, prompt, parameters=None)`

Generate streaming response from Ollama API.

**Parameters:**
- `model` (str): Name of the model to use
- `prompt` (str): Input prompt for the model
- `parameters` (dict, optional): Model parameters (temperature, max_tokens, etc.)

**Returns:** Generator yielding response chunks as dictionaries

**Example:**
```python
for chunk in client.generate_streaming("llama2", "Hello world"):
    if "response" in chunk:
        print(chunk["response"], end="")
```

##### `generate_non_streaming(model, prompt, parameters=None)`

Generate non-streaming (aggregated) response from Ollama API.

**Parameters:**
- `model` (str): Name of the model to use
- `prompt` (str): Input prompt for the model
- `parameters` (dict, optional): Model parameters

**Returns:** `OllamaResponse` object with complete aggregated text

**Example:**
```python
response = client.generate_non_streaming("llama2", "Hello world")
print(response.text)
print(f"Model: {response.model}")
print(f"Done: {response.done}")
```

##### `generate(model, prompt, stream=False, parameters=None)`

Convenience method that chooses between streaming and non-streaming based on the `stream` parameter.

**Parameters:**
- `model` (str): Name of the model to use
- `prompt` (str): Input prompt for the model
- `stream` (bool): If True, return streaming response; if False, return aggregated response
- `parameters` (dict, optional): Model parameters

**Returns:** Either `StreamingResponse` generator or `OllamaResponse` object

##### `list_models()`

List available models.

**Returns:** Dictionary containing available models

**Example:**
```python
models = client.list_models()
for model in models.get("models", []):
    print(f"Model: {model['name']}, Size: {model.get('size', 'Unknown')}")
```

##### `health_check()`

Check if Ollama API is healthy and reachable.

**Returns:** `True` if healthy, `False` otherwise

**Example:**
```python
if client.health_check():
    print("Ollama API is running")
else:
    print("Ollama API is not reachable")
```

### OllamaResponse

Response object for non-streaming responses.

#### Attributes

- `text` (str): Complete generated text
- `model` (str): Model used for generation
- `done` (bool): Whether generation is complete
- `context` (list, optional): Context for the response
- `total_duration` (int, optional): Total duration in nanoseconds
- `load_duration` (int, optional): Model load duration in nanoseconds
- `prompt_eval_count` (int, optional): Number of tokens in prompt
- `prompt_eval_duration` (int, optional): Prompt evaluation duration in nanoseconds
- `eval_count` (int, optional): Number of tokens generated
- `eval_duration` (int, optional): Generation duration in nanoseconds

## Error Handling

The wrapper provides specific exception types for different error scenarios:

### Exception Types

- `OllamaError`: Base exception for all Ollama-related errors
- `OllamaTimeoutError`: Raised when requests timeout
- `OllamaConnectionError`: Raised for connection-related errors
- `OllamaInvalidResponseError`: Raised when API returns invalid responses

### Example Error Handling

```python
from ollama_wrapper import OllamaClient, OllamaError, OllamaTimeoutError

try:
    with OllamaClient() as client:
        response = client.generate_non_streaming("llama2", "Hello")
        print(response.text)
        
except OllamaTimeoutError:
    print("Request timed out")
except OllamaConnectionError:
    print("Could not connect to Ollama API")
except OllamaError as e:
    print(f"Ollama API error: {e}")
    if hasattr(e, 'status_code'):
        print(f"Status code: {e.status_code}")
```

## Model Parameters

You can pass various parameters to control model behavior:

```python
parameters = {
    "temperature": 0.7,      # Controls randomness (0.0-1.0)
    "top_p": 0.9,           # Nucleus sampling parameter
    "top_k": 40,            # Top-k sampling parameter
    "num_predict": 100,     # Maximum number of tokens to predict
    "stop": ["\n", "END"],  # Stop sequences
    "seed": 42,             # Random seed for reproducible results
}

response = client.generate_non_streaming(
    model="llama2",
    prompt="Tell me about AI",
    parameters=parameters
)
```

## Performance Tips

1. **Use streaming for long responses**: For responses that might be lengthy, use streaming to get results as they're generated
2. **Adjust timeouts**: Set appropriate timeouts based on your model size and expected response time
3. **Use context manager**: Always use the client in a context manager or call `close()` to properly clean up resources
4. **Health check before use**: Check API health before making requests to avoid unnecessary failures

## Development

### Running Tests

```bash
cd python/ollama_wrapper
pip install -e ".[dev]"
pytest tests/
```

### Code Formatting

```bash
black src/ tests/ examples/
isort src/ tests/ examples/
```

### Type Checking

```bash
mypy src/ollama_wrapper/
```

## Examples

See the `examples/` directory for more comprehensive usage examples:

- `basic_usage.py`: Demonstrates basic streaming and non-streaming usage
- More examples coming soon...

## Requirements

- Python 3.8+
- `requests` library
- Ollama API server running (typically on `http://localhost:11434`)

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run tests and ensure they pass
6. Submit a pull request

## Support

If you encounter issues or have questions:

1. Check the [examples](examples/) for usage patterns
2. Review the API documentation above
3. Open an issue on GitHub with detailed information about your problem