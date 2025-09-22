"""
Advanced example demonstrating error handling, configuration, and utility functions.

This example shows how to use the Ollama API wrapper with proper error handling,
custom configuration, and utility functions for response processing.
"""

import sys
import time
import json
from typing import Dict, Any, List

# Add the src directory to Python path for imports
sys.path.insert(0, '../src')

from ollama_wrapper import (
    OllamaClient, 
    OllamaConfig, 
    OllamaError, 
    OllamaTimeoutError, 
    OllamaConnectionError,
    format_model_info,
    aggregate_streaming_response
)
from ollama_wrapper.config import validate_model_parameters, calculate_tokens_per_second


def create_configured_client() -> OllamaClient:
    """Create an Ollama client with custom configuration."""
    config = OllamaConfig(
        base_url="http://localhost:11434",
        timeout=45,  # Longer timeout for complex queries
        stream_timeout=90,
        retry_attempts=5,  # More retries for robustness
        retry_backoff_factor=0.5,
        default_model="llama2",
        default_parameters={
            "temperature": 0.7,
            "top_p": 0.9,
        }
    )
    
    return OllamaClient(
        base_url=config.base_url,
        timeout=config.timeout,
        stream_timeout=config.stream_timeout,
        retry_attempts=config.retry_attempts,
        retry_backoff_factor=config.retry_backoff_factor,
    )


def demonstrate_comprehensive_error_handling():
    """Demonstrate comprehensive error handling patterns."""
    print("=== Error Handling Demonstration ===")
    
    try:
        # Use a non-existent server to trigger connection error
        client = OllamaClient(base_url="http://nonexistent:11434", timeout=5)
        
        with client:
            response = client.generate_non_streaming(
                model="test-model",
                prompt="Hello world"
            )
            print(f"Response: {response.text}")
            
    except OllamaConnectionError as e:
        print(f"✓ Successfully caught connection error: {e}")
    except OllamaTimeoutError as e:
        print(f"✓ Successfully caught timeout error: {e}")
    except OllamaError as e:
        print(f"✓ Successfully caught general Ollama error: {e}")
        if hasattr(e, 'status_code') and e.status_code:
            print(f"  Status code: {e.status_code}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


def demonstrate_model_management():
    """Demonstrate model listing and information formatting."""
    print("\n=== Model Management ===")
    
    try:
        with create_configured_client() as client:
            # Check API health first
            if not client.health_check():
                print("⚠️  Ollama API is not healthy - skipping model demonstration")
                return
                
            # List and format models
            try:
                models_response = client.list_models()
                formatted_info = format_model_info(models_response)
                print(formatted_info)
                
                # Extract model names for further use
                model_names = [model['name'] for model in models_response.get('models', [])]
                if model_names:
                    print(f"\n✓ Found {len(model_names)} models")
                    return model_names[0]  # Return first model for use in other examples
                else:
                    print("⚠️  No models available")
                    return None
                    
            except OllamaError as e:
                print(f"✗ Failed to list models: {e}")
                return None
                
    except Exception as e:
        print(f"✗ Unexpected error in model management: {e}")
        return None


def demonstrate_parameter_validation():
    """Demonstrate parameter validation and sanitization."""
    print("\n=== Parameter Validation ===")
    
    # Valid parameters
    try:
        valid_params = {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 150,
            "stop": ["END", "\n"],
        }
        
        validated = validate_model_parameters(valid_params)
        print(f"✓ Valid parameters: {validated}")
        
    except ValueError as e:
        print(f"✗ Validation failed: {e}")
    
    # Invalid parameters
    invalid_cases = [
        {"temperature": 5.0},  # Too high
        {"top_p": 1.5},        # Too high
        {"top_k": -1},         # Negative
        {"invalid_type": object()},  # Invalid type
    ]
    
    for i, invalid_params in enumerate(invalid_cases, 1):
        try:
            validate_model_parameters(invalid_params)
            print(f"✗ Case {i}: Should have failed validation")
        except ValueError as e:
            print(f"✓ Case {i}: Correctly rejected - {e}")


def demonstrate_streaming_aggregation(model_name: str = "llama2"):
    """Demonstrate streaming response aggregation."""
    print(f"\n=== Streaming Aggregation (Model: {model_name}) ===")
    
    if not model_name:
        print("⚠️  No model available for streaming demonstration")
        return
    
    try:
        with create_configured_client() as client:
            
            if not client.health_check():
                print("⚠️  Ollama API is not healthy - skipping streaming demonstration")
                return
            
            prompt = "Count from 1 to 5 with descriptions."
            print(f"Prompt: {prompt}")
            
            # Manual aggregation using utility function
            print("\n--- Manual Aggregation ---")
            streaming_chunks = client.generate_streaming(
                model=model_name,
                prompt=prompt,
                parameters={
                    "temperature": 0.5,
                    "num_predict": 50,  # Limit response length for demo
                }
            )
            
            # Convert to list to use utility function
            chunks_list = list(streaming_chunks)
            aggregated_text = aggregate_streaming_response(chunks_list)
            
            print(f"Aggregated text: {aggregated_text}")
            print(f"Total chunks: {len(chunks_list)}")
            
            # Calculate performance metrics
            last_chunk = chunks_list[-1] if chunks_list else {}
            tokens_per_sec = calculate_tokens_per_second(
                last_chunk.get("eval_count"),
                last_chunk.get("eval_duration")
            )
            
            if tokens_per_sec:
                print(f"Performance: {tokens_per_sec:.2f} tokens/second")
                
            # Compare with non-streaming
            print("\n--- Non-Streaming Comparison ---")
            start_time = time.time()
            non_streaming_response = client.generate_non_streaming(
                model=model_name,
                prompt=prompt,
                parameters={
                    "temperature": 0.5,
                    "num_predict": 50,
                }
            )
            end_time = time.time()
            
            print(f"Non-streaming text: {non_streaming_response.text}")
            print(f"Non-streaming duration: {end_time - start_time:.2f} seconds")
            
            # Verify they produce similar results (allowing for randomness)
            if len(aggregated_text) > 0 and len(non_streaming_response.text) > 0:
                print("✓ Both methods produced text responses")
            
    except OllamaError as e:
        print(f"✗ Ollama error in streaming demonstration: {e}")
    except Exception as e:
        print(f"✗ Unexpected error in streaming demonstration: {e}")


def demonstrate_convenience_methods(model_name: str = "llama2"):
    """Demonstrate convenience methods and different usage patterns."""
    print(f"\n=== Convenience Methods (Model: {model_name}) ===")
    
    if not model_name:
        print("⚠️  No model available for convenience methods demonstration")
        return
        
    try:
        with create_configured_client() as client:
            
            if not client.health_check():
                print("⚠️  Ollama API is not healthy - skipping convenience demonstration")
                return
            
            prompt = "What is 2+2?"
            
            # Using the convenience generate() method
            print("--- Convenience Method: Non-streaming ---")
            response = client.generate(
                model=model_name,
                prompt=prompt,
                stream=False,
                parameters={"temperature": 0.1}  # Low temperature for consistent math
            )
            print(f"Answer: {response.text}")
            
            print("\n--- Convenience Method: Streaming ---")
            print("Answer: ", end="", flush=True)
            
            for chunk in client.generate(
                model=model_name,
                prompt=prompt,
                stream=True,
                parameters={"temperature": 0.1}
            ):
                if "response" in chunk and chunk["response"]:
                    print(chunk["response"], end="", flush=True)
                if chunk.get("done"):
                    print()  # New line after completion
                    break
                    
    except OllamaError as e:
        print(f"✗ Ollama error in convenience methods: {e}")
    except Exception as e:
        print(f"✗ Unexpected error in convenience methods: {e}")


def main():
    """Run all advanced examples."""
    print("Advanced Ollama API Wrapper Examples")
    print("=" * 50)
    
    # Run demonstrations
    demonstrate_comprehensive_error_handling()
    
    available_model = demonstrate_model_management()
    
    demonstrate_parameter_validation()
    
    demonstrate_streaming_aggregation(available_model)
    
    demonstrate_convenience_methods(available_model)
    
    print("\n" + "=" * 50)
    print("Advanced examples completed!")
    print("\nNote: Some examples require Ollama to be running with models available.")
    print("If you see connection errors, make sure Ollama is running on localhost:11434")


if __name__ == "__main__":
    main()