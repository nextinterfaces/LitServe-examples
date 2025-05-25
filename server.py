import io
import json
import logging
from typing import List, Dict, Any, Union
import requests
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import litserve as ls
from fastapi import HTTPException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageClassifier(ls.LitAPI):
    def setup(self, device: str):
        """Initialize the model and preprocessing pipeline."""
        try:
            logger.info(f"Setting up model on device: {device}")
            # Load pre-trained ResNet model
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.model.eval()
            self.model.to(device)
            
            # Load ImageNet class labels
            logger.info("Loading ImageNet labels...")
            response = requests.get("https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json")
            response.raise_for_status()
            self.labels = json.loads(response.text)
            
            # Define image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            logger.info("Setup complete")
        except Exception as e:
            logger.error(f"Error during setup: {str(e)}")
            raise

    def decode_request(self, request: Union[Dict[str, str], List[Dict[str, str]]]) -> List[str]:
        """Extract image URLs from the request."""
        try:
            logger.info(f"Decoding request: {request}")
            if isinstance(request, dict):
                if "image_url" not in request:
                    raise HTTPException(status_code=400, detail="Missing 'image_url' in request")
                return [str(request["image_url"])]
            elif isinstance(request, list):
                if not all("image_url" in r for r in request):
                    raise HTTPException(status_code=400, detail="Missing 'image_url' in one or more items")
                return [str(r["image_url"]) for r in request]
            raise HTTPException(status_code=400, detail=f"Invalid request format: {type(request)}")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error decoding request: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def preprocess(self, image_urls: List[str]) -> torch.Tensor:
        """Download and preprocess images."""
        try:
            logger.info(f"Preprocessing {len(image_urls)} images")
            processed_images = []
            for url in image_urls:
                try:
                    # Download image
                    logger.info(f"Downloading image from {url}")
                    response = requests.get(url.strip())
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content)).convert('RGB')
                    
                    # Preprocess image
                    processed = self.transform(image)
                    processed_images.append(processed)
                except requests.RequestException as e:
                    logger.error(f"Error downloading image from {url}: {str(e)}")
                    raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")
                except Exception as e:
                    logger.error(f"Error preprocessing image from {url}: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
            
            if not processed_images:
                raise HTTPException(status_code=400, detail="No images were successfully processed")
                
            return torch.stack(processed_images)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def predict(self, image_urls: List[str]) -> List[Dict[str, Any]]:
        """Run inference on the images."""
        try:
            logger.info(f"Starting prediction for {len(image_urls)} images")
            # Process each image individually
            results = []
            for url in image_urls:
                # Preprocess single image
                processed = self.preprocess([url])
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(processed)
                    probabilities = F.softmax(outputs, dim=1)
                    
                    # Get top 5 predictions
                    top5_prob, top5_idx = torch.topk(probabilities, 5)
                    
                    predictions = [
                        {
                            "label": self.labels[idx.item()],
                            "probability": float(prob)
                        }
                        for prob, idx in zip(top5_prob[0], top5_idx[0])
                    ]
                    results.append({"predictions": predictions})
            
            logger.info(f"Prediction complete, results: {results}")
            return results
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def encode_response(self, output: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format the response."""
        try:
            logger.info(f"Encoding response for {len(output)} results")
            if len(output) == 1:
                return output[0]
            return {"batch_results": output}
        except Exception as e:
            logger.error(f"Error encoding response: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Create and run the server
    logger.info("Starting LitServer...")
    classifier = ImageClassifier(max_batch_size=16)
    server = ls.LitServer(classifier, accelerator="auto")  # Use GPU if available
    server.run(port=8000) 