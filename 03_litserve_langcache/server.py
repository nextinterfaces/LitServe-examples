import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import litserve as ls
from fastapi import HTTPException
import hashlib
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticCache(ls.LitAPI):
    def setup(self, device: str):
        """Initialize the Redis LangCache Embed v1 model and cache."""
        try:
            logger.info(f"Setting up model on device: {device}")
            # Load the Redis LangCache Embed v1 model
            self.model = SentenceTransformer('redis/langcache-embed-v1')
            
            # Simple in-memory cache for demonstration
            # In production, you would use Redis or another persistent cache
            self.cache = {}
            self.embeddings_cache = {}
            
            # Cache configuration
            self.similarity_threshold = 0.85  # Threshold for cache hits
            self.max_cache_size = 1000
            
            logger.info("Model and cache initialized successfully")
        except Exception as e:
            logger.error(f"Error during setup: {str(e)}")
            raise

    def _generate_cache_key(self, text: str) -> str:
        """Generate a hash-based cache key for the input text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _find_similar_cached_query(self, query_embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """Find a similar cached query using semantic similarity."""
        if not self.embeddings_cache:
            return None
        
        # Get all cached embeddings
        cached_embeddings = []
        cache_keys = []
        
        for key, data in self.embeddings_cache.items():
            cached_embeddings.append(data['embedding'])
            cache_keys.append(key)
        
        if not cached_embeddings:
            return None
        
        # Calculate similarities
        cached_embeddings = np.array(cached_embeddings)
        similarities = cosine_similarity([query_embedding], cached_embeddings)[0]
        
        # Find the most similar cached query
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]
        
        if max_similarity >= self.similarity_threshold:
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

    def _add_to_cache(self, query: str, response: Dict[str, Any], embedding: np.ndarray):
        """Add query, response, and embedding to cache."""
        cache_key = self._generate_cache_key(query)
        
        # Manage cache size
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
            del self.embeddings_cache[oldest_key]
        
        # Add to cache
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

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate the request."""
        try:
            logger.info(f"Decoding request: {request}")
            
            if not isinstance(request, dict):
                raise HTTPException(status_code=400, detail="Request must be a JSON object")
            
            if "query" not in request:
                raise HTTPException(status_code=400, detail="Missing 'query' field in request")
            
            query = str(request["query"]).strip()
            
            if not query:
                raise HTTPException(status_code=400, detail="Query cannot be empty")
            
            # Optional parameters
            use_cache = request.get("use_cache", True)
            similarity_threshold = request.get("similarity_threshold", self.similarity_threshold)
            
            return {
                "query": query,
                "use_cache": use_cache,
                "similarity_threshold": similarity_threshold
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error decoding request: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process query with semantic caching."""
        try:
            query = inputs["query"]
            use_cache = inputs["use_cache"]
            similarity_threshold = inputs["similarity_threshold"]
            
            # Update similarity threshold if provided
            original_threshold = self.similarity_threshold
            self.similarity_threshold = similarity_threshold
            
            logger.info(f"Processing query: '{query}' (cache: {use_cache})")
            
            # Generate embedding for the query
            query_embedding = self.model.encode([query])[0]
            
            # Check cache if enabled
            cached_result = None
            if use_cache:
                cached_result = self._find_similar_cached_query(query_embedding)
            
            if cached_result:
                # Return cached result
                response = {
                    "query": query,
                    "response": "This is a simulated response based on cached data.",
                    "cache_hit": True,
                    "cached_data": cached_result,
                    "embedding_dimension": len(query_embedding),
                    "processing_time": 0.001  # Very fast for cached responses
                }
            else:
                # Simulate processing (in real scenario, this would be LLM inference)
                start_time = time.time()
                
                # Simulated response generation
                simulated_response = f"This is a simulated response for the query: '{query}'. " \
                                   f"In a real implementation, this would be generated by an LLM."
                
                processing_time = time.time() - start_time
                
                response = {
                    "query": query,
                    "response": simulated_response,
                    "cache_hit": False,
                    "embedding_dimension": len(query_embedding),
                    "processing_time": processing_time
                }
                
                # Add to cache if enabled
                if use_cache:
                    self._add_to_cache(query, response, query_embedding)
                
                logger.info(f"Generated new response. Processing time: {processing_time:.4f}s")
            
            # Restore original threshold
            self.similarity_threshold = original_threshold
            
            return response
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Format the response."""
        try:
            logger.info("Encoding response")
            return output
        except Exception as e:
            logger.error(f"Error encoding response: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Create and run the server
    logger.info("Starting LitServer for Semantic Caching with Redis LangCache Embed v1...")
    cache_api = SemanticCache()
    server = ls.LitServer(cache_api, accelerator="auto")
    server.run(port=8002)  # Use different port to avoid conflicts
