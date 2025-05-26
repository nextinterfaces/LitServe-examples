import logging
from typing import Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from redis import Redis
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FastAPI Semantic Cache with Redis Vector DB",
    description="Semantic caching service using Redis as a vector database",
    version="1.0.0"
)

# Redis configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
VECTOR_DIM = 384  # Dimension of redis/langcache-embed-v1 embeddings
INDEX_NAME = "semantic-cache"

class QueryRequest(BaseModel):
    query: str = Field(..., description="The input query text")
    use_cache: bool = Field(True, description="Whether to use the cache")
    similarity_threshold: float = Field(0.85, description="Similarity threshold for cache hits")

class CacheData(BaseModel):
    cached_response: str
    similarity_score: float
    cached_query: str
    cache_timestamp: float

class QueryResponse(BaseModel):
    query: str
    response: str
    cache_hit: bool
    cached_data: Optional[CacheData] = None
    embedding_dimension: int
    processing_time: float

class SemanticCacheService:
    def __init__(self):
        """Initialize the semantic cache service."""
        self.model = None
        self.redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.similarity_threshold = 0.85
        self.max_cache_size = 1000

    async def initialize(self, device: str = "cpu"):
        """Initialize the model and Redis vector database."""
        try:
            logger.info(f"Setting up model on device: {device}")
            self.model = SentenceTransformer('redis/langcache-embed-v1')
            
            # Create Redis search index if it doesn't exist
            try:
                # Define the index schema
                schema = (
                    VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": VECTOR_DIM, "DISTANCE_METRIC": "COSINE"}),
                    TextField("query"),
                    TextField("response"),
                    NumericField("timestamp")
                )
                
                # Create the index
                self.redis_client.ft(INDEX_NAME).create_index(
                    schema,
                    definition=IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH)
                )
                logger.info("Created Redis search index")
            except Exception as e:
                if "Index already exists" not in str(e):
                    raise
                logger.info("Redis search index already exists")
            
            logger.info("Initialization completed successfully")
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    def _find_similar_cached_query(self, query_embedding: np.ndarray, threshold: float) -> Optional[Dict[str, Any]]:
        """Find a similar cached query using Redis vector similarity search."""
        try:
            # Prepare the vector similarity query
            query_vector = query_embedding.astype(np.float32).tobytes()
            q = Query(f"*=>[KNN 1 @embedding $vec AS score]")\
                .return_fields("query", "response", "timestamp", "score")\
                .dialect(2)\
                .paging(0, 1)
            
            # Execute the query
            results = self.redis_client.ft(INDEX_NAME).search(
                q,
                {"vec": query_vector}
            )
            
            if len(results.docs) > 0 and float(results.docs[0].score) >= threshold:
                doc = results.docs[0]
                logger.info(f"Cache hit! Similarity: {float(doc.score):.4f}")
                return {
                    'cached_response': doc.response,
                    'similarity_score': float(doc.score),
                    'cached_query': doc.query,
                    'cache_timestamp': float(doc.timestamp)
                }
            
            return None
        except Exception as e:
            logger.error(f"Error in vector similarity search: {str(e)}")
            return None

    def _add_to_cache(self, query: str, response: str, embedding: np.ndarray):
        """Add query, response, and embedding to Redis vector database."""
        try:
            # Generate a unique key for the cache entry
            cache_key = f"cache:{int(time.time() * 1000)}"
            
            # Store the data in Redis
            self.redis_client.hset(
                cache_key,
                mapping={
                    "query": query,
                    "response": response,
                    "embedding": embedding.astype(np.float32).tobytes(),
                    "timestamp": str(time.time())
                }
            )
            
            # Maintain cache size limit
            cache_size = len(self.redis_client.keys("cache:*"))
            if cache_size > self.max_cache_size:
                # Remove oldest entries
                keys = sorted(self.redis_client.keys("cache:*"))
                for key in keys[:cache_size - self.max_cache_size]:
                    self.redis_client.delete(key)
            
            logger.info(f"Added to cache. Cache size: {len(self.redis_client.keys('cache:*'))}")
        except Exception as e:
            logger.error(f"Error adding to cache: {str(e)}")

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a query with semantic caching."""
        try:
            query = request.query
            use_cache = request.use_cache
            threshold = request.similarity_threshold
            
            logger.info(f"Processing query: '{query}' (cache: {use_cache})")
            
            start_time = time.time()
            query_embedding = self.model.encode([query])[0]
            
            cached_result = None
            if use_cache:
                cached_result = self._find_similar_cached_query(query_embedding, threshold)
            
            if cached_result:
                processing_time = time.time() - start_time
                return QueryResponse(
                    query=query,
                    response=cached_result['cached_response'],
                    cache_hit=True,
                    cached_data=CacheData(**cached_result),
                    embedding_dimension=len(query_embedding),
                    processing_time=processing_time
                )
            
            # Simulate response generation (replace with actual LLM in production)
            simulated_response = f"This is a simulated response for: '{query}'"
            processing_time = time.time() - start_time
            
            if use_cache:
                self._add_to_cache(query, simulated_response, query_embedding)
            
            return QueryResponse(
                query=query,
                response=simulated_response,
                cache_hit=False,
                embedding_dimension=len(query_embedding),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize service
cache_service = SemanticCacheService()

@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup."""
    await cache_service.initialize()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a query with semantic caching."""
    return await cache_service.process_query(request)

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    return {
        "cache_size": len(cache_service.redis_client.keys("cache:*")),
        "max_cache_size": cache_service.max_cache_size,
        "current_threshold": cache_service.similarity_threshold
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True) 