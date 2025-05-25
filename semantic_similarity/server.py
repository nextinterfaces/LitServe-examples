import logging
from typing import List, Dict, Any, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import litserve as ls
from fastapi import HTTPException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticSimilarity(ls.LitAPI):
    def setup(self, device: str):
        """Initialize the sentence transformer model."""
        try:
            logger.info(f"Setting up model on device: {device}")
            # Load a lightweight sentence transformer model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error during setup: {str(e)}")
            raise

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Extract query and documents from the request."""
        try:
            logger.info(f"Decoding request: {request}")
            
            if not isinstance(request, dict):
                raise HTTPException(status_code=400, detail="Request must be a JSON object")
            
            if "query" not in request:
                raise HTTPException(status_code=400, detail="Missing 'query' field in request")
            
            if "documents" not in request:
                raise HTTPException(status_code=400, detail="Missing 'documents' field in request")
            
            query = str(request["query"]).strip()
            documents = request["documents"]
            
            if not query:
                raise HTTPException(status_code=400, detail="Query cannot be empty")
            
            if not isinstance(documents, list):
                raise HTTPException(status_code=400, detail="Documents must be a list")
            
            if not documents:
                raise HTTPException(status_code=400, detail="Documents list cannot be empty")
            
            # Convert all documents to strings
            documents = [str(doc).strip() for doc in documents]
            
            # Filter out empty documents
            documents = [doc for doc in documents if doc]
            
            if not documents:
                raise HTTPException(status_code=400, detail="No valid documents found")
            
            return {"query": query, "documents": documents}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error decoding request: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate semantic similarity between query and documents."""
        try:
            query = inputs["query"]
            documents = inputs["documents"]
            
            logger.info(f"Processing query: '{query}' against {len(documents)} documents")
            
            # Generate embeddings for query and documents
            query_embedding = self.model.encode([query])
            document_embeddings = self.model.encode(documents)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, document_embeddings)[0]
            
            # Create results with similarity scores
            results = []
            for i, (doc, score) in enumerate(zip(documents, similarities)):
                results.append({
                    "document": doc,
                    "similarity_score": float(score),
                    "rank": i + 1
                })
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Update ranks after sorting
            for i, result in enumerate(results):
                result["rank"] = i + 1
            
            logger.info(f"Similarity calculation complete. Top score: {results[0]['similarity_score']:.4f}")
            
            return {
                "query": query,
                "results": results,
                "total_documents": len(documents)
            }
            
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
    logger.info("Starting LitServer for Semantic Similarity...")
    similarity_api = SemanticSimilarity()
    server = ls.LitServer(similarity_api, accelerator="auto")
    server.run(port=8001)  # Use different port to avoid conflict with image classification
