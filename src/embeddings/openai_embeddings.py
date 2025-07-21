"""
OpenAI embeddings implementation using the OpenAI API.
"""
import os
import requests
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from .embedding_factory import EmbeddingModel, EmbeddingFactory

# Load environment variables
load_dotenv()

@EmbeddingFactory.register("openai")
class OpenAIEmbeddings(EmbeddingModel):
    """
    Embedding model that uses OpenAI's embedding API.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small",**kwargs: Any):
        """
        Initialize the OpenAI embeddings model.

        Args:
            api_key: OpenAI API key. If not provided, will try to load from OPENAI_API_KEY env var
            model: OpenAI model to use for embeddings. Options include:
                  - text-embedding-3-small (1536 dimensions)
                  - text-embedding-3-large (3072 dimensions)
                  - text-embedding-ada-002 (1536 dimensions, legacy)
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")

        self._model = model
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }

        if model not in self._dimensions:
            raise ValueError(f"Unsupported model: {model}. Supported models: {list(self._dimensions.keys())}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI's API.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (as lists of floats)
        """
        embeddings = []

        # OpenAI API can handle batching, so we can send all texts at once
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }

        data = {
            "model": self._model,
            "input": texts
        }

        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            raise Exception(f"Error from OpenAI API: {response.text}")

        response_data = response.json()
        
        # Sort the embeddings by index to ensure they're in the same order as the input texts
        sorted_data = sorted(response_data["data"], key=lambda x: x["index"])
        
        # Extract the embeddings
        for item in sorted_data:
            embeddings.append(item["embedding"])

        return embeddings

    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        return self._dimensions[self._model]

    @property
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        return self._model