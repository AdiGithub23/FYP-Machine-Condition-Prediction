# fake_data/sample_raw_lookback.py

# python -m fake_data.sample_raw_lookback

# This file is for generating 1200 fake hourly mean documents and saving them to MongoDB for testing purposes.

import random
from configs.mongodb_config import get_database, workspace_id

def generate_fake_hourly_mean():
    """
    Generate a fake hourly mean document similar to MongoDB structure.
    """
    return {
        "temp_shaft_mean": round(random.uniform(25.0, 45.0), 3),
        "temp_body_mean": round(random.uniform(25.0, 45.0), 3),
        "current_mean": round(random.uniform(4.0, 7.0), 3),
        "vibration_mean": round(random.uniform(7.5, 9.5), 3),
        "num_points_used": 360,
        "machine_id": workspace_id
    }

def generate_fake_lookback(n_docs=1200):
    """
    Generate a list of n_docs fake hourly mean documents.
    """
    return [generate_fake_hourly_mean() for _ in range(n_docs)]

def save_fake_lookback_to_db(fake_docs):
    """
    Insert the fake documents into MongoDB collection 'hourly_means'.
    """
    db = get_database()
    if db:
        collection = db[f"hourly_means_{workspace_id}"]
        try:
            result = collection.insert_many(fake_docs)
            print(f"✅ Inserted {len(result.inserted_ids)} fake documents into MongoDB.")
        except Exception as e:
            print(f"❌ Error inserting fake documents: {e}")
    else:
        print("❌ Failed to connect to database.")

if __name__ == "__main__":
    # Generate 1200 fake documents
    fake_lookback = generate_fake_lookback(1200)
    print(f"✅ Generated {len(fake_lookback)} fake hourly mean documents.")
    
    # Save to MongoDB
    save_fake_lookback_to_db(fake_lookback)