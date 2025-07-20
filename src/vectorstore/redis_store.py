"""
Redis vector store implementation for storing and retrieving embeddings.
"""
import os
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RedisVectorStore:
    """
    Vector store implementation using Redis.
    """

    def __init__(
        self,
        index_name: str,
        embedding_dimension: int,
        host: Optional[str] = None,
        port: Optional[int] = None,
        password: Optional[str] = None,
        db: Optional[int] = None,
        prefix: str = "doc:",
        distance_metric: str = "COSINE"
    ):
        """
        Initialize the Redis vector store.

        Args:
            index_name: Name of the Redis index to use
            embedding_dimension: Dimension of the embedding vectors
            host: Redis host. If not provided, will try to load from REDIS_HOST env var
            port: Redis port. If not provided, will try to load from REDIS_PORT env var
            password: Redis password. If not provided, will try to load from REDIS_PASSWORD env var
            db: Redis database number. If not provided, will try to load from REDIS_DB env var
            prefix: Prefix for document keys in Redis
            distance_metric: Distance metric to use for similarity search (COSINE, IP, L2)
        """
        self.index_name = index_name
        self.embedding_dimension = embedding_dimension
        self.prefix = prefix
        self.distance_metric = distance_metric

        # Get Redis connection parameters from environment variables if not provided
        self.host = host or os.environ.get("REDIS_HOST", "localhost")
        self.port = port or int(os.environ.get("REDIS_PORT", 6379))
        self.password = password or os.environ.get("REDIS_PASSWORD", "")
        if self.password == "None":
            self.password = None
        self.db = db or int(os.environ.get("REDIS_DB", 0))

        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password,
            db=self.db,
            decode_responses=True  # Automatically decode responses to strings
        )

        # Ensure the Redis client is connected
        if not self.redis_client.ping():
            raise ConnectionError("Could not connect to Redis server")


        # Create the index if it doesn't exist
        self._create_index()

    def _create_index(self):
        """
        Create the Redis index if it doesn't exist.
        """
        # Check if index exists
        try:
            self.redis_client.ft(self.index_name).info()
            return  # Index already exists
        except:
            pass
        # Create the index
        # schema = (
        #     f"FT.CREATE {self.index_name} ON HASH PREFIX 1 {self.prefix} "
        #     f"SCHEMA content TEXT WEIGHT 1.0 "
        #     f"embedding VECTOR {self.embedding_dimension} TYPE {self.distance_metric} "  # Added TYPE keyword
        #     f"metadata TEXT"
        # )
        schema = (
            f"FT.CREATE {self.index_name} ON HASH PREFIX 1 {self.prefix} "
            f"SCHEMA content TEXT WEIGHT 1.0 "
            f"metadata TEXT "
            f"embedding TEXT"
        )
        try:
            self.redis_client.execute_command(schema)
        except redis.ResponseError as e:
            if "already exists" in str(e):
                print(f"Index {self.index_name} already exists, skipping creation.")
            else:
                raise e


    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add texts and their embeddings to the vector store.

        Args:
            texts: List of text strings
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs. If not provided, UUIDs will be generated

        Returns:
            List of document IDs
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match")

        if metadatas is None:
            metadatas = [{} for _ in texts]

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Add each document to Redis
        for i, (text, embedding, metadata, doc_id) in enumerate(zip(texts, embeddings, metadatas, ids)):
            key = f"{self.prefix}{doc_id}"

            # Convert embedding to Redis format
            embedding_bytes = json.dumps(embedding)

            # Convert metadata to JSON string
            metadata_str = json.dumps(metadata)

            # Store document in Redis
            self.redis_client.hset(
                key,
                mapping={
                    "content": text,
                    "embedding": embedding_bytes,
                    "metadata": metadata_str
                }
            )

        return ids

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar documents using the query embedding.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of tuples (text, score, metadata)
        """
        import numpy as np
        from scipy.spatial.distance import cosine

        # Build a basic query to get all documents
        query = f"*"

        # Execute the query to get all documents
        results = self.redis_client.ft(self.index_name).search(query)

        # Process results and compute similarity manually
        documents_with_scores = []

        # For debugging filter results
        total_docs = len(results.docs)
        filtered_docs = 0
        for doc in results.docs:
            text = getattr(doc, "content", "")
            metadata = json.loads(getattr(doc, "metadata", "{}"))

            # Apply metadata filter if provided
            if filter_metadata is not None:
                # Check if all filter conditions match
                match = True
                for key, value in filter_metadata.items():
                    # Convert both to strings for comparison
                    metadata_value = str(metadata.get(key, ""))
                    filter_value = str(value)

                    # Check if the filter value is a substring of the metadata value
                    if filter_value not in metadata_value:
                        match = False
                        print(f"Filter mismatch: Document metadata '{key}={metadata_value}' does not match filter '{key}={filter_value}'")
                        break

                # Skip this document if it doesn't match the filter
                if not match:
                    filtered_docs += 1
                    continue
                else:
                    print(f"Filter match: Document matches filter criteria {filter_metadata}")

            # Get the document embedding
            doc_embedding_str = getattr(doc, "embedding", "[]")
            try:
                doc_embedding = json.loads(doc_embedding_str)

                # Check if dimensions match
                if len(query_embedding) != len(doc_embedding):
                    print(f"Dimension mismatch: Query embedding dimension ({len(query_embedding)}) does not match document embedding dimension ({len(doc_embedding)})")
                    continue

                # Compute cosine similarity
                similarity = 1 - cosine(query_embedding, doc_embedding)

                documents_with_scores.append((text, similarity, metadata))
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"Error processing document embedding: {e}")
                continue

        # Sort by similarity score (descending) and take top k
        documents_with_scores.sort(key=lambda x: x[1], reverse=True)
        documents = documents_with_scores[:k]

        # Print filter summary if filtering was applied
        if filter_metadata is not None:
            matched_docs = total_docs - filtered_docs
            print(f"\nFilter Summary:")
            print(f"  Total documents: {total_docs}")
            print(f"  Documents matching filter: {matched_docs} ({(matched_docs/total_docs)*100:.1f}% of total)")
            print(f"  Documents filtered out: {filtered_docs}")
            print(f"  Filter criteria: {filter_metadata}")

        return documents

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents from the vector store.

        Args:
            ids: List of document IDs to delete
        """
        for doc_id in ids:
            key = f"{self.prefix}{doc_id}"
            self.redis_client.delete(key)

    def clear(self) -> None:
        """
        Clear all documents from the vector store with the current prefix.
        """
        # Get all keys with the current prefix
        keys = self.redis_client.keys(f"{self.prefix}*")

        # Delete all keys
        if keys:
            self.redis_client.delete(*keys)
            print(f"Cleared {len(keys)} documents from the vector store.")
