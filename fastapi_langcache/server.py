import logging
from typing import Dict, Any, Optional, List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FastAPI Semantic Cache",
    description="Semantic caching service using Redis LangCache Embed v1 model",
    version="1.0.0"
)

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
        self.cache = {}
        self.embeddings_cache = {}
        self.similarity_threshold = 0.85
        self.max_cache_size = 1000

    async def initialize(self, device: str = "cpu"):
        """Initialize the model and cache asynchronously."""
        try:
            logger.info(f"Setting up model on device: {device}")
            self.model = SentenceTransformer('redis/langcache-embed-v1')
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    def _generate_cache_key(self, text: str) -> str:
        """Generate a hash-based cache key for the input text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _find_similar_cached_query(self, query_embedding: np.ndarray, threshold: float) -> Optional[Dict[str, Any]]:
        """Find a similar cached query using semantic similarity."""
        if not self.embeddings_cache:
            return None
        
        cached_embeddings = []
        cache_keys = []
        
        for key, data in self.embeddings_cache.items():
            cached_embeddings.append(data['embedding'])
            cache_keys.append(key)
        
        if not cached_embeddings:
            return None
        
        cached_embeddings = np.array(cached_embeddings)
        similarities = cosine_similarity([query_embedding], cached_embeddings)[0]
        
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]
        
        if max_similarity >= threshold:
            cache_key = cache_keys[max_similarity_idx]
            cached_data = self.cache[cache_key]
            
            logger.info(f"Cache hit! Similarity: {max_similarity:.4f}")
            return {
                'cached_response': cached_data['response'],
                'similarity_score': float(max_similarity),
                'cached_query': cached_data['query'],
                'cache_timestamp': cached_data['timestamp']
            }
        
        return None

    def _add_to_cache(self, query: str, response: str, embedding: np.ndarray):
        """Add query, response, and embedding to cache."""
        cache_key = self._generate_cache_key(query)
        
        if len(self.cache) >= self.max_cache_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
            del self.embeddings_cache[oldest_key]
        
        self.cache[cache_key] = {
            'query': query,
            'response': response,
            'timestamp': time.time()
        }
        
        self.embeddings_cache[cache_key] = {
            'embedding': embedding,
            'timestamp': time.time()
        }
        
        logger.info(f"Added to cache. Cache size: {len(self.cache)}")

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
        "cache_size": len(cache_service.cache),
        "max_cache_size": cache_service.max_cache_size,
        "current_threshold": cache_service.similarity_threshold
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True) 