"""
RAG (Retrieval-Augmented Generation) pipeline implementation.
"""
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dotenv import load_dotenv

from ..embeddings import EmbeddingFactory, EmbeddingModel
from ..llm import LLMFactory, LLMModel
from ..vectorstore import RedisVectorStore
from ..document_loaders import PDFLoader

# Load environment variables
load_dotenv()

class RAGPipeline:
    """
    RAG pipeline for document ingestion, retrieval, and generation.
    """

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        embedding_model_name: str = "bert",
        vector_store: Optional[RedisVectorStore] = None,
        index_name: str = "rag_index",
        llm_model: Optional[LLMModel] = None,
        llm_provider: str = "claude",
        llm_model_name: str = "claude-3-haiku-20240307",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 4,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the RAG pipeline.

        Args:
            embedding_model: Embedding model instance. If not provided, one will be created
            embedding_model_name: Name of the embedding model to use if creating a new one
            vector_store: Vector store instance. If not provided, one will be created
            index_name: Name of the index to use in the vector store
            llm_model: LLM model instance. If not provided, one will be created
            llm_provider: Provider for the LLM model ("claude" or "openai")
            llm_model_name: Name of the LLM model to use if creating a new one
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            top_k: Number of similar documents to retrieve
            cache_dir: Directory to cache the downloaded models. If None, uses the default cache directory.
        """
        # Initialize embedding model
        self.embedding_model = embedding_model
        if self.embedding_model is None:
            self.embedding_model = EmbeddingFactory.create_embedding_model(
                embedding_model_name, 
                cache_dir=cache_dir
            )

        # Initialize vector store
        self.vector_store = vector_store
        if self.vector_store is None:
            self.vector_store = RedisVectorStore(
                index_name=index_name,
                embedding_dimension=self.embedding_model.dimension
            )

        # Initialize document loader
        self.document_loader = PDFLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Initialize LLM model
        self.llm_model = llm_model
        if self.llm_model is None:
            self.llm_model = LLMFactory.create_llm_model(
                llm_provider,
                model=llm_model_name
            )

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k

    def ingest_documents(self, file_paths: Union[str, Path, List[Union[str, Path]]]) -> List[str]:
        """
        Ingest documents into the vector store.

        Args:
            file_paths: Path(s) to PDF file(s) or directory containing PDF files

        Returns:
            List of document IDs
        """
        # Convert single path to list
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]

        all_documents = []
        all_ids = []

        # Process each file path
        for file_path in file_paths:
            path = Path(file_path)

            # If directory, load all PDFs in it
            if path.is_dir():
                documents = self.document_loader.load_documents(path)
                all_documents.extend(documents)
            # If file, load it
            elif path.is_file() and path.suffix.lower() == '.pdf':
                documents = self.document_loader.load_document(path)
                all_documents.extend(documents)
            else:
                print(f"Skipping {path}: Not a PDF file or directory")

        # If no documents were loaded, return empty list
        if not all_documents:
            return []

        # Get text and metadata from documents
        texts = [doc["text"] for doc in all_documents]
        metadatas = [doc["metadata"] for doc in all_documents]

        # Generate embeddings
        embeddings = self.embedding_model.get_embeddings(texts)

        # Add to vector store
        ids = self.vector_store.add_texts(texts, embeddings, metadatas)

        return ids

    def retrieve(self, query: str, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query string
            filter_metadata: Optional metadata filter

        Returns:
            List of retrieved documents with text, score, and metadata
        """
        # Generate embedding for query
        query_embedding = self.embedding_model.get_embeddings([query])[0]

        # Search vector store
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=self.top_k,
            filter_metadata=filter_metadata
        )

        # Format results
        documents = []
        for text, score, metadata in results:
            documents.append({
                "text": text,
                "score": score,
                "metadata": metadata
            })

        return documents

    def generate(self, query: str, context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a response to a query using retrieved context.

        Args:
            query: Query string
            context: Optional list of context documents. If not provided, will retrieve context

        Returns:
            Generated response
        """
        # Retrieve context if not provided
        if context is None:
            context = self.retrieve(query)

        # Format context for prompt
        context_text = ""
        for i, doc in enumerate(context):
            context_text += f"Document {i+1}:\n{doc['text']}\n\n"

        # Create prompt with clear instructions on using the context
        prompt = f"""
            You are a helpful assistant that answers questions based on the provided context.
            If you don't know the answer based on the context, just say that you don't know.
            Don't try to make up an answer.

            First, carefully read and analyze the following context documents:

            {context_text}

            Now, using ONLY the information provided in the context above, answer this question:
            {query}

            Your answer should:
            1. Be based solely on the information in the context documents
            2. Cite specific parts of the context that support your answer
            3. Be comprehensive and detailed
            4. If the context doesn't contain relevant information, acknowledge this limitation

            Answer:
        """

        # Generate response using the LLM model
        return self.llm_model.generate(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

    def query(self, query: str, filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        End-to-end RAG pipeline: retrieve and generate.

        Args:
            query: Query string
            filter_metadata: Optional metadata filter for retrieval

        Returns:
            Dictionary with query, context, and response
        """
        # Retrieve context
        context = self.retrieve(query, filter_metadata)

        # Generate response
        response = self.generate(query, context)

        return {
            "query": query,
            "context": context,
            "response": response
        }
