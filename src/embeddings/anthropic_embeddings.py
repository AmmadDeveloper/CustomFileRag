"""
Anthropic embeddings implementation using the Anthropic API.
"""
import os
import requests
import json
from typing import List, Optional
import anthropic
from dotenv import load_dotenv

from .embedding_factory import EmbeddingModel, EmbeddingFactory

# Load environment variables
load_dotenv()

@EmbeddingFactory.register("anthropic")
class AnthropicEmbeddings(EmbeddingModel):
    """
    Embedding model that uses Anthropic's embedding API.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        """
        Initialize the Anthropic embeddings model.

        Args:
            api_key: Anthropic API key. If not provided, will try to load from ANTHROPIC_API_KEY env var
            model: Anthropic model to use for embeddings
        """
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("Anthropic API key must be provided or set as ANTHROPIC_API_KEY environment variable")

        self._model = model
        self._client = anthropic.Anthropic(api_key=self._api_key)
        self._dimensions = {
            "claude-3-haiku-20240307": 1536,
            "claude-3-sonnet-20240229": 1536,
            "claude-3-opus-20240229": 3072
        }

        if model not in self._dimensions:
            raise ValueError(f"Unsupported model: {model}. Supported models: {list(self._dimensions.keys())}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Anthropic's API.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (as lists of floats)
        """
        embeddings = []

        # Process each text individually
        for text in texts:
            # Make a direct API call to the Anthropic embeddings endpoint
            headers = {
                "Content-Type": "application/json",
                "X-Api-Key": self._api_key,
                "anthropic-version": "2023-06-01"
            }

            data = {
                "model": self._model,
                "input": text
            }

            response = requests.post(
                "https://api.anthropic.com/v1/embeddings",
                headers=headers,
                json=data
            )

            if response.status_code != 200:
                raise Exception(f"Error from Anthropic API: {response.text}")

            response_data = response.json()
            embeddings.append(response_data["embedding"])

        return embeddings

    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        return self._dimensions[self._model]

    @property
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        return self._model
