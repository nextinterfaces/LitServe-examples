#!/usr/bin/env python3
"""
Very simple test for the semantic similarity API.
"""

import requests
import json

def main():
    """Test the semantic similarity API with a minimal example."""
    
    url = "http://127.0.0.1:8001/predict"
    
    # Simple test data
    data = {
        "query": "cat",
        "documents": [
            "A small furry animal that meows",
            "Computer programming language",
            "Dog is man's best friend"
        ]
    }
    
    try:
        print("Testing semantic similarity...")
        print(f"Query: {data['query']}")
        print(f"Documents: {data['documents']}")
        print()
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        
        print("Results:")
        for item in result["results"]:
            print(f"  {item['rank']}. {item['document']} (score: {item['similarity_score']:.3f})")
        
        print("\n✅ Test passed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to server. Make sure the server is running on port 8001")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
