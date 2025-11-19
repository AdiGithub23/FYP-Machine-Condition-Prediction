# test_files/test_pipelines.py

# python -m test_files.test_pipelines

# Quick test for the complete pipeline: generate fake data, store, retrieve lookback, run inference.

import random
from datetime import datetime
from configs.mongodb_config import get_database, workspace_id
from services.inference_service import InferenceService

def generate_fake_hourly_mean():
    """
    Generate a fake hourly mean document (similar to sample_raw_lookback.py).
    """
    return {
        "temp_shaft_mean": round(random.uniform(25.0, 45.0), 3),
        "temp_body_mean": round(random.uniform(25.0, 45.0), 3),
        "current_mean": round(random.uniform(4.0, 7.0), 3),
        "vibration_mean": round(random.uniform(7.5, 9.5), 3),
        "num_points_used": 360,
        "machine_id": workspace_id,
        "timestamp": datetime.now().isoformat()
    }

def generate_fake_lookback(n_docs=1200):
    """
    Generate a list of n_docs fake hourly mean documents.
    """
    return [generate_fake_hourly_mean() for _ in range(n_docs)]

def test_pipeline():
    """
    Test the complete pipeline: insert fake data, retrieve lookback, run inference.
    """
    db = get_database()
    if not db:
        print("[Test] Failed to connect to database.")
        return
    
    collection = db[f"hourly_means_{workspace_id}"]
    
    # Step 1: Generate and insert 1200 fake docs (for quick testing)
    print("[Test] Generating and inserting 1200 fake hourly mean documents...")
    fake_docs = generate_fake_lookback(1200)
    try:
        result = collection.insert_many(fake_docs)
        print(f"[Test] Inserted {len(result.inserted_ids)} fake documents into 'hourly_means_{workspace_id}'.")
    except Exception as e:
        print(f"[Test] Error inserting fake documents: {e}")
        return
    
    # Step 2: Retrieve the last 1200 lookback
    print("[Test] Retrieving last 1200 lookback...")
    lookback = list(collection.find({"machine_id": workspace_id}).sort("_id", -1).limit(1200))
    print(f"[Test] Retrieved {len(lookback)} documents for lookback.")
    
    if len(lookback) < 1200:
        print(f"[Test] Warning: Only {len(lookback)} docs retrieved. Inference may skip.")
    
    # Step 3: Feed into inference_service.py
    print("[Test] Running inference...")
    inference_service = InferenceService()
    inference_service.run_inference(lookback)
    print("[Test] Inference complete. Check logs for forecast, anomaly, alerts, and emails.")

if __name__ == "__main__":
    test_pipeline()