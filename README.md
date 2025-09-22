# glass
A Cloudways Monitoring Client

## Go Monitoring Client

The main Glass application is a Go-based monitoring client that collects system metrics including:

- CPU information and usage statistics
- Memory usage and availability
- Disk usage statistics
- Network connection information

### Building

```bash
make build
```

This will create a `bin/cloudways` executable.

### Running

```bash
./bin/cloudways
```

## Python Ollama API Wrapper

This repository also includes a Python wrapper for the Ollama API that supports both streaming and non-streaming responses.

### Features

- **Dual Mode Support**: Both streaming and non-streaming API calls
- **Response Aggregation**: Automatically aggregates streaming word-by-word responses
- **Error Handling**: Comprehensive error handling for API failures
- **Configurable Timeouts**: Separate timeout configurations for different request types
- **Type Safety**: Full type annotation support

### Quick Start

```python
from python.ollama_wrapper.src.ollama_wrapper import OllamaClient

# Non-streaming response
with OllamaClient() as client:
    response = client.generate_non_streaming(
        model="llama2",
        prompt="What is the capital of France?"
    )
    print(response.text)

# Streaming response  
with OllamaClient() as client:
    for chunk in client.generate_streaming(
        model="llama2", 
        prompt="Tell me a story"
    ):
        if "response" in chunk:
            print(chunk["response"], end="", flush=True)
```

### Installation

```bash
cd python/ollama_wrapper
pip install -e .
```

For detailed documentation, see [python/ollama_wrapper/README.md](python/ollama_wrapper/README.md).

### Requirements

- Python 3.8+
- `requests` library
- Ollama API server running (typically on `http://localhost:11434`)

## License

MIT License - see [LICENSE](LICENSE) file for details.
