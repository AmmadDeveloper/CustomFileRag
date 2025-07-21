"""
Test script for LLM models.

This script demonstrates how to use the different LLM models:
1. Claude (default)
2. OpenAI (o1-nano)

Usage:
1. Set up your environment variables in .env file
2. Run this script: python test_llm.py
"""
import os
from dotenv import load_dotenv
from src.llm import LLMFactory

# Load environment variables
load_dotenv()

def main():
    # Sample prompt for testing
    sample_prompt = "Explain the concept of Retrieval-Augmented Generation (RAG) in 3 sentences."
    
    # Test different LLM models
    models_to_test = [
        ("claude", "claude-3-haiku-20240307"),
        ("openai", "o1-preview")
    ]
    
    for provider, model_name in models_to_test:
        print(f"\n=== Testing {provider.upper()} LLM ({model_name}) ===")
        try:
            # Create LLM model
            llm_model = LLMFactory.create_llm_model(
                provider, 
                model=model_name
            )
            
            # Get model information
            print(f"Model: {llm_model.model_name}")
            
            # Generate text
            print("Generating response...")
            response = llm_model.generate(
                prompt=sample_prompt,
                max_tokens=100,
                temperature=0.7
            )
            
            # Print response
            print("\nResponse:")
            print(response)
            
            print(f"{provider.upper()} LLM test: SUCCESS")
        except Exception as e:
            print(f"{provider.upper()} LLM test: FAILED")
            print(f"Error: {str(e)}")
    
    # List all available models
    print("\n=== Available LLM Models ===")
    available_models = LLMFactory.list_available_models()
    print(f"Available models: {available_models}")

if __name__ == "__main__":
    main()