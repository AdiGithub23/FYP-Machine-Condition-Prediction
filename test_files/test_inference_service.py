# test_inference_service.py
# Run this script to test inference_service.py: python test_files/test_inference_service.py

from services.inference_service import InferenceService
from configs.mongodb_config import get_database

def test_inference_service():
    # Step 1: Retrieve last 1200 documents (same as notebook Cell 2)
    db = get_database()
    if not db:
        print("Database connection failed. Cannot test.")
        return
    
    collection = db["hourly_means"]
    retrieved_docs = list(collection.find().sort("_id", -1).limit(1200))
    retrieved_docs = retrieved_docs[::-1]
    
    if len(retrieved_docs) < 1200:
        print(f"Not enough data: {len(retrieved_docs)} documents (need 1200). Populate MongoDB first.")
        return
    
    print(f"Retrieved {len(retrieved_docs)} documents for testing.")
    
    # Step 2: Instantiate and run inference
    inference = InferenceService()
    forecast, alerts = inference.run_inference(retrieved_docs)
    
    # Step 3: Print results
    if forecast is not None:
        print(f"Forecast shape: {forecast.shape}")
        print(f"Alerts: {alerts}")
    else:
        print("Inference failed. Check logs.")


if __name__ == "__main__":
    test_inference_service()


