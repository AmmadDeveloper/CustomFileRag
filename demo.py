"""
Demo script for the RAG pipeline.

This script demonstrates how to use the RAG pipeline to:
1. Ingest PDF documents
2. Query the knowledge base
3. Get responses based on the document content

Usage:
1. Set up your environment variables in .env file
2. Place PDF files in the 'data' directory
3. Run this script: python demo.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv

from src.rag import RAGPipeline

# Load environment variables
load_dotenv()

def main():
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag_pipeline = RAGPipeline(
        # Using E5 embeddings instead of BERT for better performance
        # E5 is a state-of-the-art open-source embedding model from Microsoft
        # that outperforms BERT and MiniLM on various benchmarks
        embedding_model_name="e5",
        index_name="demo_e5",
        llm_model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.7,
        chunk_size=1000,
        chunk_overlap=200,
        top_k=100
    )

    # Check if there are PDF files in the data directory
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {data_dir}. Please add some PDF files and run again.")
        return

    # Ingest documents
    print(f"Ingesting {len(pdf_files)} PDF documents...")
    doc_ids = rag_pipeline.ingest_documents(data_dir)
    print(f"Ingested {len(doc_ids)} document chunks.")

    # Interactive query loop
    print("\nRAG Demo - Ask questions about your documents")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ["exit", "quit", "q"]:
            break


        use_filter = input("Do you want to filter by metadata? (y/n): ").lower() == 'y'
        filter_metadata = None

        if use_filter:
            filter_type = input("Filter by (1) filename or (2) source path? (1/2): ")
            filter_value = input("Enter filter value: ")

            if filter_type == "1":
                filter_metadata = {"filename": filter_value}
                print(f"Filtering by filename: {filter_value}")
            elif filter_type == "2":
                filter_metadata = {"source": filter_value}
                print(f"Filtering by source: {filter_value}")
            else:
                print("Invalid filter type. No filter will be applied.")

        print("\nRetrieving and generating response...")
        result = rag_pipeline.query(query, filter_metadata)

        print("\n=== Retrieved Context ===")
        for i, doc in enumerate(result["context"]):
            print(f"Document {i+1} (Score: {doc['score']:.4f}):")
            print(f"Source: {doc['metadata'].get('source', 'Unknown')}")
            print(f"Filename: {doc['metadata'].get('filename', 'Unknown')}")
            print(f"Text: {doc['text'][:100]}...")
            print()

        print("=== Generated Response ===")
        print(result["response"])

if __name__ == "__main__":
    main()
