# Image Classification with LitServe

This example demonstrates how to use LitServe to serve a machine learning model for image classification. The example uses a pre-trained ResNet model from torchvision to classify images from URLs.

## Features

- **Fast inference** with batching support
- **GPU acceleration** when available (MPS on Mac, CUDA on Linux/Windows)
- **OpenAPI documentation** automatically generated
- **Health check endpoint** for monitoring
- **Request/response validation** with proper error handling
- **Batch processing** support for multiple images
- **ImageNet classification** with 1000+ classes

## How it works

1. **Input**: You provide image URL(s) in JSON format
2. **Download**: Images are downloaded and preprocessed
3. **Inference**: ResNet model classifies the images
4. **Output**: Returns top 5 predictions with confidence scores

## Setup

1. Create and activate virtual environment (if not already done):
```bash
# From the repository root
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Navigate to the image classification example:
```bash
cd image_classification/
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

Start the server:
```bash
# Make sure virtual environment is activated
source ../venv/bin/activate  # On Windows: ..\venv\Scripts\activate
python3 server.py
```

The server will be available at http://localhost:8000

## API Usage

### Endpoints
- `POST /predict` - Image classification endpoint
- `GET /health` - Health check endpoint
- `GET /docs` - OpenAPI documentation (Swagger UI)

### Request Format

**Single image:**
```json
{
  "image_url": "https://example.com/image.jpg"
}
```

**Multiple images (batch):**
```json
[
  {"image_url": "https://example.com/image1.jpg"},
  {"image_url": "https://example.com/image2.jpg"}
]
```

### Response Format

**Single image response:**
```json
{
  "predictions": [
    {
      "label": "golden retriever",
      "probability": 0.8234
    },
    {
      "label": "labrador retriever", 
      "probability": 0.1456
    }
  ]
}
```

**Batch response:**
```json
{
  "batch_results": [
    {
      "predictions": [
        {
          "label": "golden retriever",
          "probability": 0.8234
        }
      ]
    }
  ]
}
```

## Example Usage

### Using curl
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/American_Eskimo_Dog.jpg/320px-American_Eskimo_Dog.jpg"}'
```

### Using the provided client
```bash
# Make sure virtual environment is activated and you're in image_classification/
source ../venv/bin/activate  # On Windows: ..\venv\Scripts\activate
python3 test_client.py
```

## Model Information

This example uses:
- **Model**: ResNet-18 pre-trained on ImageNet
- **Classes**: 1000 ImageNet classes
- **Input size**: 224x224 pixels
- **Preprocessing**: Standard ImageNet normalization

## Performance

- **Batch processing**: Supports up to 16 images per batch
- **GPU acceleration**: Automatically uses available GPU (MPS/CUDA)
- **Memory efficient**: Processes images individually to handle large batches

## Error Handling

The API includes comprehensive error handling for:
- Invalid image URLs
- Network timeouts
- Unsupported image formats
- Server errors
- Malformed requests

## Customization

You can easily customize this example:

1. **Different model**: Replace ResNet with other torchvision models
2. **Custom classes**: Use your own trained model
3. **Preprocessing**: Modify image preprocessing pipeline
4. **Batch size**: Adjust `max_batch_size` parameter
5. **Response format**: Customize the output structure
