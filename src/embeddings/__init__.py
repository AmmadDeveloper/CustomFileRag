# Import models to register them with the factory
from .embedding_factory import EmbeddingModel, EmbeddingFactory
from .anthropic_embeddings import AnthropicEmbeddings
from .bert_embeddings import BERTEmbeddings
from .e5_embeddings import E5Embeddings
from .openai_embeddings import OpenAIEmbeddings

# Export the factory and model classes
__all__ = ['EmbeddingModel', 'EmbeddingFactory', 'AnthropicEmbeddings', 'BERTEmbeddings', 'E5Embeddings', 'OpenAIEmbeddings']
