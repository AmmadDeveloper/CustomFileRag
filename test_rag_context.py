import os
from src.rag import RAGPipeline
from src.vectorstore import RedisVectorStore
from src.embeddings import EmbeddingFactory

# Sample documents to add to the vector store
sample_docs = [
    "The capital of France is Paris. Paris is known for the Eiffel Tower and the Louvre Museum.",
    "The capital of Japan is Tokyo. Tokyo is the most populous metropolitan area in the world.",
    "The capital of Italy is Rome. Rome is known for the Colosseum and Vatican City.",
    "The capital of Spain is Madrid. Madrid is known for its elegant boulevards and expansive parks."
]

def test_rag_context():
    print("Initializing RAG pipeline...")
    # Initialize RAG pipeline with BERT embeddings (more consistent dimensions)
    rag = RAGPipeline(embedding_model_name="bert")

    # Clear the vector store to start fresh
    print("Clearing existing documents from vector store...")
    rag.vector_store.clear()

    print("Adding sample documents to vector store...")
    # Get embeddings for sample documents
    embeddings = rag.embedding_model.get_embeddings(sample_docs)

    # Add documents to vector store
    metadatas = [{"source": f"sample_doc_{i}"} for i in range(len(sample_docs))]
    rag.vector_store.add_texts(sample_docs, embeddings, metadatas)

    print("\nTesting query with context...")
    # Test query that requires context
    query = "What is the capital of France?"

    # First, retrieve the context
    context = rag.retrieve(query)
    print("\nRetrieved context:")
    for i, doc in enumerate(context):
        print(f"Document {i+1} (score: {doc['score']:.4f}):")
        print(doc['text'])
        print()

    # Generate response using the context
    print("\nGenerating response...")
    response = rag.generate(query, context)

    print("\nQuery:", query)
    print("\nResponse:")
    print(response)

    # Test another query
    query2 = "What is Tokyo known for?"
    print("\n\nTesting another query:", query2)

    # Generate response
    result = rag.query(query2)

    print("\nRetrieved context:")
    for i, doc in enumerate(result["context"]):
        print(f"Document {i+1} (score: {doc['score']:.4f}):")
        print(doc['text'])
        print()

    print("\nResponse:")
    print(result["response"])

if __name__ == "__main__":
    test_rag_context()
