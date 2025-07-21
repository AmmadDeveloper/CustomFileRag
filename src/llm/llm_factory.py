"""
Factory pattern implementation for LLM models.
This allows for easy extension to support multiple LLM providers.
"""
from abc import ABC, abstractmethod
import os
from typing import List, Dict, Any, Optional

class LLMModel(ABC):
    """Abstract base class for LLM models."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: The prompt to generate text from
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (higher = more creative)
            
        Returns:
            Generated text
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the LLM model."""
        pass


class LLMFactory:
    """Factory class for creating LLM model instances."""
    
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
    def create_llm_model(cls, model_name: str, **kwargs) -> LLMModel:
        """
        Create an instance of the specified LLM model.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            An instance of the specified LLM model
            
        Raises:
            ValueError: If the model name is not registered
        """
        if model_name not in cls._models:
            raise ValueError(f"Unknown LLM model: {model_name}")
        
        return cls._models[model_name](**kwargs)
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """
        List all registered LLM models.
        
        Returns:
            List of registered model names
        """
        return list(cls._models.keys())