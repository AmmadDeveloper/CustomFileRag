"""
Demo script for the RAG pipeline.

This script demonstrates how to use the RAG pipeline to:
1. Ingest PDF documents
2. Query the knowledge base
3. Get responses based on the document content
4. Use different LLM models (Claude or OpenAI)

Usage:
1. Set up your environment variables in .env file
2. Place PDF files in the 'data' directory
3. Run this script: python demo.py
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from src.rag import RAGPipeline
from src.llm import LLMFactory

# Load environment variables
load_dotenv()

def main():
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Determine which LLM provider to use
    if len(sys.argv) > 1 and sys.argv[1].lower() == "openai":
        llm_provider = "openai"
        llm_model_name = "gpt-4.1-nano"  # OpenAI's o1-nano model
        print(f"""Using OpenAI {llm_model_name} model""")
    else:
        llm_provider = "claude"
        llm_model_name = "claude-3-haiku-20240307"
        print("Using Claude model (default)")

    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag_pipeline = RAGPipeline(
        # Using OpenAI embeddings
        embedding_model_name="openai",
        index_name=f"demo_{llm_provider}",
        # Using the factory pattern for LLM models
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        max_tokens=700,
        temperature=0.5,
        chunk_size=2000,
        chunk_overlap=200,
        top_k=5
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
