"""
BERT embeddings implementation using the Hugging Face transformers library.
"""
import os
import pathlib
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

from .embedding_factory import EmbeddingModel, EmbeddingFactory

# Load environment variables
load_dotenv()

# Define the cache directory for storing models locally
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model_cache")

@EmbeddingFactory.register("bert")
class BERTEmbeddings(EmbeddingModel):
    """
    Embedding model that uses BERT models from Hugging Face.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize the BERT embeddings model.

        Args:
            model_name: Name of the BERT model to use from Hugging Face
            device: Device to run the model on ('cpu' or 'cuda'). If None, will use CUDA if available.
            cache_dir: Directory to cache the downloaded models. If None, uses the default cache directory.
        """
        self._model_name = model_name

        # Set device
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        # Set up cache directory
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        os.makedirs(self._cache_dir, exist_ok=True)

        # Load tokenizer and model from cache if available, otherwise download
        print(f"Loading model {model_name} (using cache directory: {self._cache_dir})")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self._cache_dir)
        self._model = AutoModel.from_pretrained(model_name, cache_dir=self._cache_dir).to(self._device)

        # Set model to evaluation mode
        self._model.eval()

        # Set embedding dimension based on the model
        with torch.no_grad():
            # Get a sample embedding to determine the dimension
            inputs = self._tokenizer("Sample text", return_tensors="pt").to(self._device)
            outputs = self._model(**inputs)
            self._dimension = outputs.last_hidden_state.size(-1)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using BERT.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (as lists of floats)
        """
        embeddings = []

        # Process texts in batches to avoid memory issues
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenize and get model inputs
            inputs = self._tokenizer(batch_texts, padding=True, truncation=True, 
                                    return_tensors="pt", max_length=512).to(self._device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self._model(**inputs)

                # Use mean pooling to get a single vector per text
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state

                # Calculate mean of token embeddings, considering only non-padding tokens
                for j in range(len(batch_texts)):
                    # Get the embeddings for this text
                    text_embedding = token_embeddings[j]
                    text_mask = attention_mask[j]

                    # Calculate mean, ignoring padding tokens
                    sum_embeddings = torch.sum(text_embedding * text_mask.unsqueeze(-1), dim=0)
                    sum_mask = torch.sum(text_mask)
                    mean_embedding = sum_embeddings / sum_mask

                    # Convert to list of floats and add to results
                    embeddings.append(mean_embedding.cpu().tolist())

        return embeddings

    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        return self._model_name
