import os
import time
from src.embeddings.embedding_factory import EmbeddingFactory

def test_e5_caching():
    print("Testing E5 embeddings caching...")
    
    # Create the first instance of E5Embeddings
    print("\nCreating first instance (may download model if not cached):")
    start_time = time.time()
    e5_embeddings1 = EmbeddingFactory.create_embedding_model("e5")
    load_time1 = time.time() - start_time
    print(f"First load time: {load_time1:.2f} seconds")
    
    # Generate embeddings for a sample text
    sample_texts = ["This is a test sentence to generate embeddings for."]
    embeddings = e5_embeddings1.get_embeddings(sample_texts)
    print(f"Generated embeddings with dimension: {len(embeddings[0])}")
    
    # Create a second instance of E5Embeddings (should load from cache)
    print("\nCreating second instance (should load from cache):")
    start_time = time.time()
    e5_embeddings2 = EmbeddingFactory.create_embedding_model("e5")
    load_time2 = time.time() - start_time
    print(f"Second load time: {load_time2:.2f} seconds")
    
    # Compare load times
    print(f"\nLoad time comparison:")
    print(f"First load: {load_time1:.2f} seconds")
    print(f"Second load: {load_time2:.2f} seconds")
    print(f"Speedup: {load_time1/load_time2:.2f}x faster")
    
    return load_time1, load_time2

def test_bert_caching():
    print("\nTesting BERT embeddings caching...")
    
    # Create the first instance of BERTEmbeddings
    print("\nCreating first instance (may download model if not cached):")
    start_time = time.time()
    bert_embeddings1 = EmbeddingFactory.create_embedding_model("bert")
    load_time1 = time.time() - start_time
    print(f"First load time: {load_time1:.2f} seconds")
    
    # Generate embeddings for a sample text
    sample_texts = ["This is a test sentence to generate embeddings for."]
    embeddings = bert_embeddings1.get_embeddings(sample_texts)
    print(f"Generated embeddings with dimension: {len(embeddings[0])}")
    
    # Create a second instance of BERTEmbeddings (should load from cache)
    print("\nCreating second instance (should load from cache):")
    start_time = time.time()
    bert_embeddings2 = EmbeddingFactory.create_embedding_model("bert")
    load_time2 = time.time() - start_time
    print(f"Second load time: {load_time2:.2f} seconds")
    
    # Compare load times
    print(f"\nLoad time comparison:")
    print(f"First load: {load_time1:.2f} seconds")
    print(f"Second load: {load_time2:.2f} seconds")
    print(f"Speedup: {load_time1/load_time2:.2f}x faster")
    
    return load_time1, load_time2

if __name__ == "__main__":
    # Test E5 embeddings caching
    e5_load_time1, e5_load_time2 = test_e5_caching()
    
    # Test BERT embeddings caching
    bert_load_time1, bert_load_time2 = test_bert_caching()
    
    # Print summary
    print("\nSummary:")
    print(f"E5 first load: {e5_load_time1:.2f} seconds, second load: {e5_load_time2:.2f} seconds, speedup: {e5_load_time1/e5_load_time2:.2f}x")
    print(f"BERT first load: {bert_load_time1:.2f} seconds, second load: {bert_load_time2:.2f} seconds, speedup: {bert_load_time1/bert_load_time2:.2f}x")
    
    # Check if model_cache directory exists and contains files
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
    if os.path.exists(cache_dir):
        print(f"\nModel cache directory exists at: {cache_dir}")
        files = os.listdir(cache_dir)
        print(f"Cache directory contains {len(files)} files/directories")
    else:
        print(f"\nModel cache directory does not exist at: {cache_dir}")