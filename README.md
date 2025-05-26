# LitServe Examples - Sentence Transformer Applications

This repository showcases different implementations of Sentence Transformer apps.

### Common Setup for All Examples

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Navigate to the example you want to try:
```bash
cd <example_folder>/
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the server:
```bash
python3 server.py
```

5. Test with the provided client:
```bash
python3 client.py
```

## Technologies Used

### Server Frameworks
- **FastAPI**
- **LitServe**: A lightweight serving framework for ML models. 

### Caching Solutions
- **Semantic Cache**: Implemented to:
  - Reduce redundant model computations
  - Handle semantically similar queries
  - Improve response times for similar questions
  - Save computational resources

- **Redis Vector Database**: Used as an advanced caching solution with:
  - HNSW (Hierarchical Navigable Small World) indexing for fast similarity search
  - Efficient vector storage and retrieval
  - Scalable and persistent storage
  - Real-time similarity matching

## Models Used
- **redis/langcache-embed-v1**: A sentence transformer model optimized for:
  - Semantic text embeddings
  - Cache-friendly representations
  - Fast similarity computations
  - Consistent vector dimensions (384)

## Example Structure

1. **01_image_classification**: Basic image classification example
2. **02_semantic_similarity**: Simple semantic similarity matching
3. **03_litserve_langcache**: LitServe with basic caching
4. **04_fastapi_langcache**: FastAPI with in-memory semantic cache
5. **05_fastapi_langcache_redis_vectordb**: Advanced implementation using Redis vector DB
6. **06_litserve_langcache_redis**: LitServe with Redis integration

## Getting Started

Each example directory contains its own README with specific setup instructions. Generally, you'll need:

1. Python 3.8+
2. Redis 7.0+ with RediSearch module (for vector DB examples)
3. Virtual environment setup
4. Required Python packages (see individual requirements.txt files)

## Performance Considerations

The examples demonstrate different approaches to caching and serving, each with its own trade-offs:

- LitServe: Simplified deployment but less customizable
- FastAPI: More flexible but requires more setup