"""
Factory pattern implementation for embedding models.
This allows for easy extension to support multiple embedding providers.
"""
from abc import ABC, abstractmethod
import os
from typing import List, Dict, Any, Optional

class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (as lists of floats)
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        pass


class EmbeddingFactory:
    """Factory class for creating embedding model instances."""
    
    _models: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a model class with the factory.
        
        Args:
            name: Name to register the model under
        """
        def inner_wrapper(wrapped_class):
            cls._models[name] = wrapped_class
            return wrapped_class
        return inner_wrapper
    
    @classmethod
    def create_embedding_model(cls, model_name: str, **kwargs) -> EmbeddingModel:
        """
        Create an instance of the specified embedding model.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            An instance of the specified embedding model
            
        Raises:
            ValueError: If the model name is not registered
        """
        if model_name not in cls._models:
            raise ValueError(f"Unknown embedding model: {model_name}")
        
        return cls._models[model_name](**kwargs)
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """
        List all registered embedding models.
        
        Returns:
            List of registered model names
        """
        return list(cls._models.keys())