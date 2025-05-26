# Semantic Caching with Redis LangCache Embed v1

This example demonstrates semantic caching using the Redis LangCache Embed v1 model with LitServe. The API intelligently caches responses based on semantic similarity, allowing similar queries to retrieve cached results instead of reprocessing.

## Features

- **Semantic Caching**: Uses the Redis LangCache Embed v1 model to find semantically similar cached queries
- **Intelligent Cache Hits**: Configurable similarity threshold for determining cache hits
- **Fast Retrieval**: Cached responses are returned instantly without reprocessing
- **Memory Management**: Automatic cache size management with FIFO eviction
- **Flexible Configuration**: Adjustable similarity thresholds and cache settings
- **Production Ready**: Designed for real-world semantic caching scenarios

## How it works

1. **Query Processing**: Input queries are converted to 768-dimensional embeddings using Redis LangCache Embed v1
2. **Similarity Search**: The system searches cached embeddings for semantically similar queries
3. **Cache Decision**: If similarity exceeds the threshold (default 0.85), returns cached response
4. **Response Generation**: For cache misses, generates new response and caches it
5. **Cache Management**: Automatically manages cache size and evicts old entries

## Model Information

This example uses the **Redis LangCache Embed v1** model:
- **Base Model**: Alibaba-NLP/gte-modernbert-base (ModernBERT)
- **Training Data**: Fine-tuned on Quora question pairs dataset
- **Embedding Dimension**: 768
- **Max Sequence Length**: 8192 tokens
- **Similarity Function**: Cosine similarity
- **Performance**: 90% accuracy on Quora similarity tasks

## Setup

1. Create and activate virtual environment (if not already done):
```bash
# From the repository root
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Navigate to the langcache_embed example:
```bash
cd langcache_embed/
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

Start the server:
```bash
# Make sure virtual environment is activated
python3 server.py
```

The server will be available at http://localhost:8002

## API Usage

### Endpoint
- `POST /predict` - Process query with semantic caching

### Request Format
```json
{
  "query": "What is machine learning?",
  "use_cache": true,
  "similarity_threshold": 0.85
}
```

### Parameters
- `query` (required): The input query text
- `use_cache` (optional): Enable/disable caching (default: true)
- `similarity_threshold` (optional): Similarity threshold for cache hits (default: 0.85)

### Response Format
```json
{
  "query": "What is machine learning?",
  "response": "Generated or cached response text",
  "cache_hit": false,
  "embedding_dimension": 768,
  "processing_time": 0.1234,
  "cached_data": {
    "cached_response": "Previous response",
    "similarity_score": 0.92,
    "cached_query": "Can you explain machine learning?",
    "cache_timestamp": 1640995200.0
  }
}
```

## Example Usage

### Using curl
```bash
# First query (cache miss)
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?"
  }'

# Similar query (cache hit)
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can you explain machine learning?"
  }'
```

### Using the provided client
```bash
# Make sure virtual environment is activated and you're in langcache_embed/
source ../venv/bin/activate  # On Windows: ..\venv\Scripts\activate
python3 client.py

# Or run the simple test
python3 test_cache.py
```

## Example Scenarios

### Cache Hit Example
```
Query 1: "What is machine learning?"
→ Cache Miss, generates response, caches embedding

Query 2: "Can you explain machine learning?"
→ Cache Hit (similarity: 0.92), returns cached response instantly
```

### Similarity Threshold Testing
```
Base Query: "What is artificial intelligence?"
Test Query: "Explain AI to me"

Threshold 0.95: Cache Miss (similarity: 0.87)
Threshold 0.85: Cache Hit (similarity: 0.87)
```

## Use Cases

- **LLM Response Caching**: Cache expensive LLM responses for similar questions
- **FAQ Systems**: Instantly serve answers to frequently asked questions
- **Customer Support**: Reduce response time for common inquiries
- **Knowledge Bases**: Efficient retrieval of similar information
- **Chatbots**: Improve response speed with semantic caching
- **Search Optimization**: Cache search results for similar queries

## Configuration

### Cache Settings
- **Similarity Threshold**: 0.85 (adjustable per request)
- **Max Cache Size**: 1000 entries
- **Eviction Policy**: FIFO (First In, First Out)
- **Embedding Model**: redis/langcache-embed-v1

### Performance Tuning
- **Lower Threshold**: More cache hits, potentially less precise
- **Higher Threshold**: Fewer cache hits, more precise matching
- **Cache Size**: Larger cache = more memory, better hit rates

## Production Considerations

### Redis Integration
In production, replace the in-memory cache with Redis:
```python
import redis
r = redis.Redis(host='localhost', port=6379, db=0)
```

### Monitoring
- Track cache hit rates
- Monitor similarity score distributions
- Measure response time improvements

### Scaling
- Use Redis Cluster for distributed caching
- Implement cache warming strategies
- Consider embedding precomputation for static content

## Customization

You can easily customize this example:

1. **Different Models**: Replace with other sentence transformer models
2. **Cache Backend**: Integrate with Redis, Memcached, or other cache systems
3. **Similarity Metrics**: Experiment with different similarity functions
4. **Cache Policies**: Implement LRU, TTL, or other eviction strategies
5. **Preprocessing**: Add text normalization or cleaning steps
