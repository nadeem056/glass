"""
Example usage of the Ollama API wrapper.

This script demonstrates both streaming and non-streaming modes of the OllamaClient.
"""

import sys
import time
from typing import Dict, Any

# Add the src directory to Python path for imports
sys.path.insert(0, '../src')

from ollama_wrapper import OllamaClient, OllamaError


def example_non_streaming():
    """Example of non-streaming (aggregated) response."""
    print("=== Non-Streaming Example ===")
    
    try:
        with OllamaClient(base_url="http://localhost:11434", timeout=30) as client:
            # Check if API is healthy
            if not client.health_check():
                print("Warning: Ollama API might not be running")
                return
            
            # List available models
            try:
                models = client.list_models()
                print(f"Available models: {[model['name'] for model in models.get('models', [])]}")
            except OllamaError as e:
                print(f"Could not fetch models: {e}")
            
            # Generate non-streaming response
            prompt = "What is the capital of France?"
            print(f"Prompt: {prompt}")
            
            start_time = time.time()
            response = client.generate_non_streaming(
                model="llama2",  # Change this to an available model
                prompt=prompt,
                parameters={
                    "temperature": 0.7,
                    "max_tokens": 100,
                }
            )
            end_time = time.time()
            
            print(f"\nComplete Response:")
            print(f"Text: {response.text}")
            print(f"Model: {response.model}")
            print(f"Done: {response.done}")
            print(f"Duration: {end_time - start_time:.2f} seconds")
            
            if response.eval_count and response.eval_duration:
                tokens_per_second = response.eval_count / (response.eval_duration / 1_000_000_000)
                print(f"Tokens per second: {tokens_per_second:.2f}")
    
    except OllamaError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def example_streaming():
    """Example of streaming response."""
    print("\n=== Streaming Example ===")
    
    try:
        with OllamaClient(base_url="http://localhost:11434", stream_timeout=60) as client:
            # Check if API is healthy
            if not client.health_check():
                print("Warning: Ollama API might not be running")
                return
            
            prompt = "Tell me a short story about a robot."
            print(f"Prompt: {prompt}")
            print("\nStreaming Response:")
            
            aggregated_text = ""
            chunk_count = 0
            start_time = time.time()
            
            for chunk in client.generate_streaming(
                model="llama2",  # Change this to an available model
                prompt=prompt,
                parameters={
                    "temperature": 0.8,
                    "max_tokens": 200,
                }
            ):
                chunk_count += 1
                
                if "response" in chunk:
                    text_chunk = chunk["response"]
                    aggregated_text += text_chunk
                    print(text_chunk, end="", flush=True)
                
                # Print final statistics when done
                if chunk.get("done", False):
                    end_time = time.time()
                    print(f"\n\n--- Streaming Statistics ---")
                    print(f"Total chunks: {chunk_count}")
                    print(f"Total text length: {len(aggregated_text)}")
                    print(f"Duration: {end_time - start_time:.2f} seconds")
                    
                    if chunk.get("eval_count") and chunk.get("eval_duration"):
                        tokens_per_second = chunk["eval_count"] / (chunk["eval_duration"] / 1_000_000_000)
                        print(f"Tokens per second: {tokens_per_second:.2f}")
    
    except OllamaError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def example_convenience_method():
    """Example using the convenience generate() method."""
    print("\n=== Convenience Method Example ===")
    
    try:
        with OllamaClient() as client:
            if not client.health_check():
                print("Warning: Ollama API might not be running")
                return
            
            prompt = "What is machine learning?"
            
            # Non-streaming using convenience method
            print("Non-streaming response:")
            response = client.generate(
                model="llama2",
                prompt=prompt,
                stream=False,
                parameters={"temperature": 0.5}
            )
            print(f"Response: {response.text[:100]}...")  # First 100 chars
            
            print("\nStreaming response:")
            # Streaming using convenience method
            for chunk in client.generate(
                model="llama2",
                prompt=prompt,
                stream=True,
                parameters={"temperature": 0.5}
            ):
                if "response" in chunk and chunk["response"]:
                    print(chunk["response"], end="", flush=True)
                if chunk.get("done"):
                    print("\n")
                    break
    
    except OllamaError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    """Run all examples."""
    print("Ollama API Wrapper Examples")
    print("=" * 40)
    
    # Run examples
    example_non_streaming()
    example_streaming()
    example_convenience_method()
    
    print("\n" + "=" * 40)
    print("Examples completed!")
    print("\nNote: Make sure Ollama is running and you have the 'llama2' model")
    print("available, or change the model name in the examples to match your setup.")


if __name__ == "__main__":
    main()