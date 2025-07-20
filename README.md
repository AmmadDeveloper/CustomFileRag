# RAG with E5, BERT and Claude

A Retrieval-Augmented Generation (RAG) system using E5 or BERT for embeddings, Anthropic's Claude for generation, and Redis as a vector database.

## Features

- **Factory Pattern for Embedding Models**: Easily extend with additional embedding providers
- **E5 Embeddings**: State-of-the-art open-source embeddings from Microsoft that outperform BERT
- **BERT Embeddings**: High-quality embeddings from Hugging Face BERT models
- **Anthropic Embeddings**: Alternative embeddings from Claude models
- **Local Model Caching**: Save embedding models locally to prevent re-downloading
- **Redis Vector Database**: Fast similarity search using Redis as a vector store
- **PDF Processing**: Extract and chunk text from PDF documents
- **RAG Pipeline**: Complete pipeline for document ingestion, retrieval, and generation

## Project Structure

```
.
├── src/
│   ├── embeddings/
│   │   ├── embedding_factory.py  # Factory pattern for embedding models
│   │   ├── e5_embeddings.py  # E5 embeddings implementation (recommended)
│   │   ├── bert_embeddings.py  # BERT embeddings implementation
│   │   └── anthropic_embeddings.py  # Anthropic embeddings implementation
│   ├── vectorstore/
│   │   └── redis_store.py  # Redis vector store implementation
│   ├── document_loaders/
│   │   └── pdf_loader.py  # PDF document loader
│   └── rag/
│       └── rag_pipeline.py  # RAG pipeline implementation
├── data/  # Directory for PDF files
├── demo.py  # Demo script
├── requirements.txt  # Project dependencies
└── .env.example  # Example environment variables
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-claude-redis.git
cd rag-claude-redis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env .env
```

4. Edit the `.env` file with your API keys and Redis configuration.

## Usage

### 1. Add PDF Documents

Place your PDF files in the `data` directory.

### 2. Run the Demo

```bash
python demo.py
```

This will:
- Ingest all PDF files in the `data` directory
- Start an interactive query loop where you can ask questions about the documents

### 3. Using the RAG Pipeline in Your Code

```python
from src.rag import RAGPipeline

# Initialize the pipeline with E5 embeddings (recommended)
rag = RAGPipeline(embedding_model_name="e5")

# To use BERT embeddings instead
# rag = RAGPipeline(embedding_model_name="bert")

# To use Anthropic embeddings instead
# rag = RAGPipeline(embedding_model_name="anthropic")

# Ingest documents
doc_ids = rag.ingest_documents("path/to/your/pdfs")

# Query the knowledge base
result = rag.query("What is the main topic of the document?")
print(result["response"])
```

## Extending the System

### Adding New Embedding Models

1. Create a new embedding model class that inherits from `EmbeddingModel`
2. Register it with the factory using the `@EmbeddingFactory.register` decorator
3. Implement the required methods: `get_embeddings`, `dimension`, and `model_name`

Example:

```python
from src.embeddings import EmbeddingModel, EmbeddingFactory

@EmbeddingFactory.register("my_model")
class MyEmbeddingModel(EmbeddingModel):
    def __init__(self, model_param: str = "default_value"):
        self._model_param = model_param
        self._dimension = 768  # Set the dimension of your embeddings

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Implement your embedding logic here
        # Return a list of embedding vectors (as lists of floats)
        embeddings = []
        for text in texts:
            # Generate embedding for text
            embedding = [0.0] * self._dimension  # Replace with actual embedding
            embeddings.append(embedding)
        return embeddings

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return f"my_model_{self._model_param}"
```

You can then use your custom embedding model:

```python
from src.rag import RAGPipeline

# Initialize the pipeline with your custom embedding model
rag = RAGPipeline(embedding_model_name="my_model", model_param="custom_value")
```

## Model Caching

The system now supports local caching of embedding models to prevent re-downloading them in future sessions. This is particularly useful for:

- Reducing bandwidth usage
- Speeding up initialization time
- Working in environments with limited internet connectivity
- Ensuring consistent model versions

### How It Works

When you initialize an E5 or BERT embedding model, it will:
1. Check if the model is already cached in the local cache directory
2. If found, load the model from the cache
3. If not found, download the model and save it to the cache

### Default Cache Location

By default, models are cached in a `model_cache` directory at the root of the project.

### Specifying a Custom Cache Directory

You can specify a custom cache directory when initializing the RAG pipeline:

```python
from src.rag import RAGPipeline

# Use a custom cache directory
rag = RAGPipeline(
    embedding_model_name="e5",
    cache_dir="/path/to/your/custom/cache"
)
```

Or when directly creating an embedding model:

```python
from src.embeddings import EmbeddingFactory

embeddings = EmbeddingFactory.create_embedding_model(
    "e5",
    cache_dir="/path/to/your/custom/cache"
)
```

## Requirements

- Python 3.8+
- Redis with RediSearch module
- PyTorch and Transformers libraries for E5 and BERT embeddings
- Anthropic API key (only needed if using Anthropic embeddings)

## Embedding Models

The system supports the following embedding models:

1. **E5** (Recommended): State-of-the-art open-source embeddings from Microsoft
   - Better performance than BERT/MiniLM on various benchmarks
   - Available in different sizes (small, base, large)
   - Default model: `intfloat/e5-large-v2`

2. **BERT**: Traditional embedding model using Hugging Face transformers
   - Default model: `sentence-transformers/all-MiniLM-L6-v2`

3. **Anthropic**: Proprietary embeddings from Anthropic's Claude models
   - Requires an Anthropic API key

## License

MIT
