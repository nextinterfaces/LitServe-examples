# Semantic Similarity Search with LitServe

This example demonstrates how to build a simple semantic similarity search API using LitServe and sentence transformers. The API allows you to find the most semantically similar documents to a given query.

## Features

- **Simple API**: Send a query and list of documents, get back ranked similarity scores
- **Fast inference**: Uses the lightweight `all-MiniLM-L6-v2` model for quick embeddings
- **Cosine similarity**: Calculates semantic similarity using cosine similarity
- **Ranked results**: Returns documents sorted by similarity score
- **Easy to use**: Simple JSON API with clear request/response format

## How it works

1. **Input**: You provide a query text and a list of documents
2. **Embedding**: The model converts both query and documents into vector embeddings
3. **Similarity**: Cosine similarity is calculated between query and each document
4. **Ranking**: Documents are ranked by similarity score (highest first)
5. **Output**: Returns ranked list with similarity scores

## Setup

1. Create and activate virtual environment (if not already done):
```bash
# From the repository root
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Navigate to the semantic similarity example:
```bash
cd semantic_similarity/
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

The server will be available at http://localhost:8001

## API Usage

### Endpoint
- `POST /predict` - Find similar documents

### Request Format
```json
{
  "query": "your search query",
  "documents": [
    "document 1 text",
    "document 2 text",
    "document 3 text"
  ]
}
```

### Response Format
```json
{
  "query": "your search query",
  "results": [
    {
      "document": "most similar document",
      "similarity_score": 0.8542,
      "rank": 1
    },
    {
      "document": "second most similar document",
      "similarity_score": 0.7231,
      "rank": 2
    }
  ],
  "total_documents": 2
}
```

## Example Usage

### Using curl
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "documents": [
      "Deep learning uses neural networks for pattern recognition",
      "The weather is sunny today",
      "Supervised learning requires labeled training data"
    ]
  }'
```

### Using the provided client
```bash
# Make sure virtual environment is activated and you're in semantic_similarity/
source ../venv/bin/activate  # On Windows: ..\venv\Scripts\activate
python3 client.py

# Or run the simple test
python3 test_simple.py
```

## Example Output

For the query "machine learning algorithms" with various documents, you might get:

```
Results (ranked by similarity):
------------------------------------------------------------
Rank 1: 0.7234
Document: Deep learning uses neural networks for pattern recognition

Rank 2: 0.6891
Document: Supervised learning requires labeled training data

Rank 3: 0.1234
Document: The weather is sunny today
```

## Use Cases

- **Document search**: Find relevant documents in a knowledge base
- **Content recommendation**: Recommend similar articles or posts
- **FAQ matching**: Match user questions to existing FAQ entries
- **Product search**: Find similar products based on descriptions
- **Research**: Find related papers or articles

## Model Information

This example uses the `all-MiniLM-L6-v2` sentence transformer model:
- **Size**: ~23MB (very lightweight)
- **Speed**: Fast inference, suitable for real-time applications
- **Quality**: Good balance of speed and accuracy for general text similarity
- **Languages**: Primarily English, but works reasonably well for other languages

## Customization

You can easily customize this example:

1. **Different model**: Change the model in `server.py` to use other sentence transformers
2. **Similarity metric**: Replace cosine similarity with other metrics
3. **Preprocessing**: Add text cleaning or preprocessing steps
4. **Filtering**: Add minimum similarity thresholds
5. **Caching**: Add embedding caching for frequently used documents
