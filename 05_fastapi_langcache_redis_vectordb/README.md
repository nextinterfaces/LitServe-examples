# FastAPI Semantic Cache with Redis Vector Database

This example demonstrates how to implement a semantic caching system using FastAPI and Redis as a vector database. The system uses Redis's vector similarity search capabilities to find semantically similar queries in the cache.

## Features

- FastAPI server with semantic caching
- Redis vector database for storing and searching embeddings
- Sentence transformer model for generating embeddings
- Asynchronous client with rich console output
- Configurable similarity threshold
- Cache size management

## Prerequisites

- Python 3.8+
- Redis 7.0+ with RediSearch module
- Virtual environment (recommended)

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Install and start Redis with RediSearch module:

Using Docker (recommended):
```bash
docker run -p 6379:6379 redis/redis-stack-server:latest
```

Or install Redis Stack locally following the instructions at: https://redis.io/docs/stack/get-started/install/

## Usage

1. Start the server:
```bash
python server.py
```

2. In a separate terminal, run the client demo:
```bash
python client.py
```

The demo will:
1. Check server health
2. Send multiple test queries
3. Show cache hits and misses
4. Display similarity scores for cached results
5. Show processing times

## API Endpoints

- `GET /health` - Check server health
- `GET /cache/stats` - Get cache statistics
- `POST /query` - Send a query with optional caching parameters

## Configuration

Server configuration (in `server.py`):
- `REDIS_HOST` - Redis server host (default: 'localhost')
- `REDIS_PORT` - Redis server port (default: 6379)
- `VECTOR_DIM` - Embedding dimension (default: 384)
- `MAX_CACHE_SIZE` - Maximum number of cached items (default: 1000)
- `DEFAULT_SIMILARITY_THRESHOLD` - Minimum similarity score for cache hits (default: 0.85)

Client configuration (in `client.py`):
- `API_URL` - Server URL (default: 'http://localhost:8000')

## How It Works

1. When a query is received:
   - The server generates an embedding using the sentence transformer model
   - It searches for similar queries in Redis using vector similarity search
   - If a similar query is found (similarity > threshold), returns the cached response
   - Otherwise, generates a new response and caches it with the query embedding

2. Redis vector similarity search:
   - Uses HNSW (Hierarchical Navigable Small World) index for efficient similarity search
   - Stores embeddings as binary data in Redis hashes
   - Uses cosine similarity as the distance metric

3. Cache management:
   - Maintains a maximum cache size
   - Removes oldest entries when the cache is full
   - Uses timestamp-based keys for easy ordering and cleanup

## Example Queries

The demo includes several test queries that demonstrate semantic similarity:

```python
test_queries = [
    "What is the capital of France?",
    "Tell me about Paris, the capital city of France",
    "What's the main city in France?",
    "How do I make a chocolate cake?",
    "What's the recipe for chocolate cake?",
    "Tell me the steps to bake a chocolate cake",
]
```

You should see cache hits for semantically similar queries, even when they are worded differently.

## Notes

- This is a demonstration example. In a production environment, you would:
  - Replace the simulated response with an actual LLM call
  - Add proper error handling and retries
  - Implement authentication and rate limiting
  - Configure Redis persistence and replication
  - Optimize Redis index parameters for your use case
  - Add monitoring and logging

- The similarity threshold can be adjusted based on your needs:
  - Higher values (e.g., 0.9) for stricter matching
  - Lower values (e.g., 0.8) for more lenient matching 