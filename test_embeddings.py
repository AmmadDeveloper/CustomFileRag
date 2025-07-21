"""
Test script for embedding models.

This script demonstrates how to use the different embedding models:
1. E5 (default)
2. BERT
3. OpenAI (new implementation)

Usage:
1. Set up your environment variables in .env file
2. Run this script: python test_embeddings.py
"""
import os
from dotenv import load_dotenv
from src.embeddings import EmbeddingFactory

# Load environment variables
load_dotenv()

def main():
    # Sample text for embedding
    sample_texts = [
        "This is a sample text to test embeddings.",
        "Another example to ensure the models work correctly."
    ]
    
    # Test different embedding models
    models_to_test = ["e5", "bert", "openai"]
    
    for model_name in models_to_test:
        print(f"\n=== Testing {model_name.upper()} Embeddings ===")
        try:
            # Create embedding model
            if model_name == "openai":
                # For OpenAI, specify the model
                embedding_model = EmbeddingFactory.create_embedding_model(
                    model_name, 
                    model="text-embedding-3-small"
                )
            else:
                embedding_model = EmbeddingFactory.create_embedding_model(model_name)
            
            # Get model information
            print(f"Model: {embedding_model.model_name}")
            print(f"Embedding dimension: {embedding_model.dimension}")
            
            # Generate embeddings
            print("Generating embeddings...")
            embeddings = embedding_model.get_embeddings(sample_texts)
            
            # Print embedding statistics
            print(f"Generated {len(embeddings)} embeddings")
            print(f"Embedding shape: {len(embeddings[0])} dimensions")
            
            # Print a small sample of the first embedding
            print(f"Sample of first embedding: {embeddings[0][:5]}...")
            
            print(f"{model_name.upper()} Embeddings test: SUCCESS")
        except Exception as e:
            print(f"{model_name.upper()} Embeddings test: FAILED")
            print(f"Error: {str(e)}")
    
    # List all available models
    print("\n=== Available Embedding Models ===")
    available_models = EmbeddingFactory.list_available_models()
    print(f"Available models: {available_models}")

if __name__ == "__main__":
    main()