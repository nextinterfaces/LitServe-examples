#!/usr/bin/env python3
"""
Simple client example for the semantic similarity search API.
"""

import requests
import json

def test_semantic_similarity():
    """Test the semantic similarity API with example data."""
    
    # API endpoint
    url = "http://127.0.0.1:8001/predict"
    
    # Example data
    query = "machine learning algorithms"
    documents = [
        "Deep learning is a subset of machine learning that uses neural networks.",
        "Python is a popular programming language for data science.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Supervised learning uses labeled data to train models.",
        "The weather is nice today and I want to go for a walk.",
        "Artificial intelligence is transforming many industries.",
        "Data preprocessing is an important step in machine learning pipelines."
    ]
    
    # Prepare request
    payload = {
        "query": query,
        "documents": documents
    }
    
    try:
        print(f"Query: '{query}'")
        print(f"Searching through {len(documents)} documents...\n")
        
        # Send request
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        print("Results (ranked by similarity):")
        print("-" * 60)
        
        for item in result["results"]:
            print(f"Rank {item['rank']}: {item['similarity_score']:.4f}")
            print(f"Document: {item['document']}")
            print()
        
        print(f"Total documents processed: {result['total_documents']}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
    except json.JSONDecodeError as e:
        print(f"Error parsing response: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def test_simple_query():
    """Test with a simple query."""
    
    url = "http://127.0.0.1:8001/predict"
    
    payload = {
        "query": "cooking recipes",
        "documents": [
            "How to bake chocolate chip cookies",
            "Machine learning model training",
            "Italian pasta recipes",
            "Software development best practices",
            "Healthy breakfast ideas"
        ]
    }
    
    try:
        print("\n" + "="*60)
        print("SIMPLE QUERY TEST")
        print("="*60)
        print(f"Query: '{payload['query']}'")
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        print("\nTop 3 results:")
        for item in result["results"][:3]:
            print(f"{item['rank']}. {item['document']} (score: {item['similarity_score']:.4f})")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Testing Semantic Similarity Search API")
    print("="*60)
    
    # Test 1: Comprehensive example
    test_semantic_similarity()
    
    # Test 2: Simple example
    test_simple_query()
