#!/usr/bin/env python3
"""
Client for testing the Redis LangCache Embed v1 semantic caching API.
"""

import requests
import json
import time
from typing import Dict, Any

class SemanticCacheClient:
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.predict_url = f"{base_url}/predict"
    
    def query(self, query: str, use_cache: bool = True, similarity_threshold: float = 0.85) -> Dict[str, Any]:
        """Send a query to the semantic cache API."""
        payload = {
            "query": query,
            "use_cache": use_cache,
            "similarity_threshold": similarity_threshold
        }
        
        try:
            response = requests.post(self.predict_url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            return {}
    
    def demonstrate_caching(self):
        """Demonstrate the semantic caching functionality."""
        print("=== Redis LangCache Embed v1 Semantic Caching Demo ===\n")
        
        # Test queries - some similar, some different
        test_queries = [
            "What is machine learning?",
            "Can you explain machine learning?",  # Similar to first
            "What is artificial intelligence?",
            "How does deep learning work?",
            "What are neural networks?",
            "Explain deep learning algorithms",  # Similar to fourth
            "What is the weather like today?",  # Different topic
            "What is machine learning about?",  # Similar to first
        ]
        
        print("Testing semantic caching with various queries...\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"Query {i}: {query}")
            print("-" * 60)
            
            # First request (should miss cache or hit if similar)
            start_time = time.time()
            result = self.query(query)
            end_time = time.time()
            
            if result:
                cache_status = "HIT" if result.get("cache_hit", False) else "MISS"
                processing_time = result.get("processing_time", 0)
                
                print(f"Cache Status: {cache_status}")
                print(f"Response: {result.get('response', 'No response')[:100]}...")
                print(f"Processing Time: {processing_time:.4f}s")
                print(f"Total Request Time: {end_time - start_time:.4f}s")
                print(f"Embedding Dimension: {result.get('embedding_dimension', 'Unknown')}")
                
                if result.get("cache_hit") and "cached_data" in result:
                    cached_data = result["cached_data"]
                    print(f"Cached Query: {cached_data.get('cached_query', 'Unknown')}")
                    print(f"Similarity Score: {cached_data.get('similarity_score', 0):.4f}")
                
                print()
            else:
                print("Failed to get response\n")
            
            # Small delay between requests
            time.sleep(0.5)
    
    def test_similarity_thresholds(self):
        """Test different similarity thresholds."""
        print("\n=== Testing Different Similarity Thresholds ===\n")
        
        base_query = "What is machine learning?"
        similar_query = "Can you explain machine learning?"
        
        # First, cache the base query
        print(f"Caching base query: {base_query}")
        self.query(base_query)
        print()
        
        # Test with different thresholds
        thresholds = [0.95, 0.85, 0.75, 0.65]
        
        for threshold in thresholds:
            print(f"Testing with threshold {threshold}:")
            result = self.query(similar_query, similarity_threshold=threshold)
            
            if result:
                cache_status = "HIT" if result.get("cache_hit", False) else "MISS"
                print(f"  Cache Status: {cache_status}")
                
                if result.get("cache_hit") and "cached_data" in result:
                    similarity = result["cached_data"].get("similarity_score", 0)
                    print(f"  Similarity Score: {similarity:.4f}")
                
                print()

def main():
    """Main function to run the client demo."""
    client = SemanticCacheClient()
    
    # Check if server is running
    try:
        response = requests.get(f"{client.base_url}/health", timeout=5)
        print("Server is running!")
    except requests.exceptions.RequestException:
        print("Error: Server is not running. Please start the server first:")
        print("python3 server.py")
        return
    
    # Run demonstrations
    client.demonstrate_caching()
    client.test_similarity_thresholds()
    
    print("Demo completed!")

if __name__ == "__main__":
    main()
