# LitServe Examples

This repository contains examples demonstrating how to use LitServe to serve different types of machine learning models.

## Examples

### 1. Image Classification (`image_classification/`)
Demonstrates image classification using a pre-trained ResNet model from torchvision.

**Features:**
- Fast inference with batching support
- GPU acceleration when available
- OpenAPI documentation
- Health check endpoint
- Request/response validation

**Quick Start:**
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Navigate to example
cd image_classification/

# Install dependencies
pip install -r requirements.txt

# Start server
python3 server.py

# Test with client
python3 test_client.py
```

Server runs on http://localhost:8000

### 2. Semantic Similarity Search (`semantic_similarity/`)
Demonstrates text semantic similarity search using sentence transformers.

**Features:**
- Simple JSON API for text similarity
- Fast lightweight model (all-MiniLM-L6-v2)
- Ranked similarity results
- Cosine similarity calculation

**Quick Start:**
```bash
# Create and activate virtual environment (if not already done)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Navigate to example
cd semantic_similarity/

# Install dependencies
pip install -r requirements.txt

# Start server
python3 server.py

# Test with client
python3 client.py
```

Server runs on http://localhost:8001

### 3. Semantic Caching with Redis LangCache Embed v1 (`langcache_embed/`)
Demonstrates intelligent semantic caching using the Redis LangCache Embed v1 model.

**Features:**
- Semantic caching based on query similarity
- Redis LangCache Embed v1 model (768-dimensional embeddings)
- Configurable similarity thresholds
- Automatic cache management with FIFO eviction
- Fast cache hit responses
- Production-ready caching patterns

**Quick Start:**
```bash
# Create and activate virtual environment (if not already done)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Navigate to example
cd langcache_embed/

# Install dependencies
pip install -r requirements.txt

# Start server
python3 server.py

# Test with client
python3 client.py
```

Server runs on http://localhost:8002

## General Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Choose an example and follow its specific setup instructions above.