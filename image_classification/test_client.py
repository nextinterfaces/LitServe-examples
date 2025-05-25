import requests
import json

def test_single_image():
    """Test classification of a single image."""
    url = "http://localhost:8000/predict"
    data = {
        "image_url": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    }
    
    try:
        print("\nSingle Image Classification:")
        print("---------------------------")
        print("Request data:", json.dumps(data, indent=2))
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        print("Raw response:", json.dumps(result, indent=2))
        if "predictions" in result:
            for pred in result["predictions"]:
                print(f"{pred['label']}: {pred['probability']:.4f}")
        else:
            print("Unexpected response format")
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print("Error response:", json.dumps(e.response.json(), indent=2))
            except ValueError:
                print("Error response:", e.response.text)

def test_batch():
    """Test batch classification of multiple images."""
    url = "http://localhost:8000/predict"
    data = [
        {"image_url": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"},
        {"image_url": "https://raw.githubusercontent.com/pytorch/hub/master/images/cat_224.jpg"}
    ]
    
    try:
        print("\nBatch Classification:")
        print("-------------------")
        print("Request data:", json.dumps(data, indent=2))
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        print("Raw response:", json.dumps(result, indent=2))
        if "batch_results" in result:
            for i, batch_result in enumerate(result["batch_results"]):
                print(f"\nImage {i + 1}:")
                if "predictions" in batch_result:
                    for pred in batch_result["predictions"]:
                        print(f"{pred['label']}: {pred['probability']:.4f}")
                else:
                    print("Unexpected batch result format")
        else:
            print("Unexpected response format")
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print("Error response:", json.dumps(e.response.json(), indent=2))
            except ValueError:
                print("Error response:", e.response.text)

if __name__ == "__main__":
    print("Testing LitServer Image Classification API")
    print("========================================")
    
    test_single_image()
    test_batch() 