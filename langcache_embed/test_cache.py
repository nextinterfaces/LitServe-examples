#!/usr/bin/env python3
"""
Simple test script for the Redis LangCache Embed v1 semantic caching API.
"""

import requests
import json
import time

def test_basic_functionality():
    """Test basic caching functionality."""
    base_url = "http://localhost:8002"
    predict_url = f"{base_url}/predict"
    
    print("=== Basic Semantic Caching Test ===\n")
    
    # Test data: pairs of similar questions
    test_cases = [
        {
            "original": "What is the capital of France?",
            "similar": "What's the capital city of France?",
            "different": "What is the weather like today?"
        },
        {
            "original": "How do I learn Python programming?",
            "similar": "What's the best way to learn Python?",
            "different": "What is quantum computing?"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"Original: {case['original']}")
        print(f"Similar: {case['similar']}")
        print(f"Different: {case['different']}")
        print("-" * 50)
        
        # 1. Send original query (should be cache miss)
        print("1. Sending original query...")
        response1 = requests.post(predict_url, json={"query": case["original"]})
        if response1.status_code == 200:
            result1 = response1.json()
            print(f"   Cache Hit: {result1.get('cache_hit', False)}")
            print(f"   Processing Time: {result1.get('processing_time', 0):.4f}s")
        
        # 2. Send similar query (should be cache hit)
        print("2. Sending similar query...")
        response2 = requests.post(predict_url, json={"query": case["similar"]})
        if response2.status_code == 200:
            result2 = response2.json()
            print(f"   Cache Hit: {result2.get('cache_hit', False)}")
            if result2.get('cache_hit') and 'cached_data' in result2:
                similarity = result2['cached_data'].get('similarity_score', 0)
                print(f"   Similarity Score: {similarity:.4f}")
        
        # 3. Send different query (should be cache miss)
        print("3. Sending different query...")
        response3 = requests.post(predict_url, json={"query": case["different"]})
        if response3.status_code == 200:
            result3 = response3.json()
            print(f"   Cache Hit: {result3.get('cache_hit', False)}")
            print(f"   Processing Time: {result3.get('processing_time', 0):.4f}s")
        
        print("\n" + "="*60 + "\n")

def test_cache_disabled():
    """Test with caching disabled."""
    base_url = "http://localhost:8002"
    predict_url = f"{base_url}/predict"
    
    print("=== Cache Disabled Test ===\n")
    
    query = "What is artificial intelligence?"
    
    # Send same query twice with cache disabled
    for i in range(2):
        print(f"Request {i+1} (cache disabled):")
        response = requests.post(predict_url, json={
            "query": query,
            "use_cache": False
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Cache Hit: {result.get('cache_hit', False)}")
            print(f"   Processing Time: {result.get('processing_time', 0):.4f}s")
        print()

def main():
    """Run all tests."""
    try:
        # Check if server is running
        response = requests.get("http://localhost:8002/health", timeout=5)
        print("Server is running! Starting tests...\n")
    except requests.exceptions.RequestException:
        print("Error: Server is not running. Please start the server first:")
        print("python3 server.py")
        return
    
    test_basic_functionality()
    test_cache_disabled()
    
    print("All tests completed!")

if __name__ == "__main__":
    main()
