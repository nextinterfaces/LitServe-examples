# LitServe Examples

This repository contains examples demonstrating how to use LitServe and related technologies to serve different types of machine learning models and implement various ML-powered features.

## Repository Structure

Each folder in this repository is a self-contained example with its own setup instructions and dependencies. Below is an overview of each example:

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

5. Test with the provided client (if available):
```bash
python3 client.py  # or test_client.py
```

### Available Examples

#### 1. Image Classification (`image_classification/`)
A demonstration of image classification using a pre-trained ResNet model from torchvision.

**Features:**
- Fast inference with batching support
- GPU acceleration when available
- OpenAPI documentation
- Health check endpoint
- Request/response validation

Server runs on http://localhost:8000

#### 2. Semantic Similarity Search (`semantic_similarity/`)
Implementation of text semantic similarity search using sentence transformers.

**Features:**
- Simple JSON API for text similarity
- Fast lightweight model (all-MiniLM-L6-v2)
- Ranked similarity results
- Cosine similarity calculation

Server runs on http://localhost:8001

#### 3. LitServe LangCache (`litserve_langcache/`)
Demonstrates intelligent semantic caching using LitServe with LangCache integration.

**Features:**
- Semantic caching based on query similarity
- Efficient embedding-based caching
- Configurable similarity thresholds
- Production-ready caching patterns

#### 4. FastAPI LangCache (`fastapi_langcache/`)
Shows how to implement semantic caching in a FastAPI application using LangCache.

**Features:**
- FastAPI integration with LangCache
- Semantic similarity-based caching
- RESTful API design
- Easy-to-follow example for FastAPI users

## Getting Started

1. Clone this repository
2. Choose an example you want to try
3. Follow the common setup instructions above
4. Check the example's specific README or documentation for additional details

## Requirements

- Python 3.7+
- Additional requirements are specified in each example's `requirements.txt`

## Contributing

Feel free to contribute additional examples or improvements to existing ones through pull requests.

## License

See individual example directories for specific licensing information.