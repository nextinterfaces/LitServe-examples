# LitServer Image Classification Example

This example demonstrates how to use LitServer to serve a machine learning model for image classification. The example uses a pre-trained ResNet model from torchvision to classify images.

## Features

- Fast inference with batching support
- GPU acceleration when available
- OpenAPI documentation
- Health check endpoint
- Request/response validation

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

Start the server with:
```bash
python server.py
```

The server will be available at http://localhost:8000

## API Endpoints

- `/predict` - POST endpoint for image classification
- `/health` - Health check endpoint
- `/docs` - OpenAPI documentation

## Example Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg"}'
```