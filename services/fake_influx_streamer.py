# services/fake_influx_streamer.py

import time
from collections import deque
from fake_data.data_generator import generate_fake_point
from services.statistics_service import StatisticsService
from configs.mongodb_config import get_database

class FakeInfluxStreamer:
    def __init__(self, interval_seconds=10, max_points=360):
        self.latest_point = None
        self.buffer = deque(maxlen=max_points)
        self.interval = interval_seconds
        
        self.stats_service = StatisticsService()
        self.point_counter = 0
        self.db = get_database()
        self.collection = self.db["hourly_means"] if self.db else None
        self.last_lookback = []

    def start_stream(self):
        """
        Continuously generate a new data point every 3 seconds.
        Every 15 seconds (after 5 points), compute mean of last 5 points and add to means_list.
        """
        while True:
            new_point = generate_fake_point()
            self.latest_point = new_point
            self.buffer.append(new_point)
            self.point_counter += 1
            print("[FakeInflux] New data point generated:", self.latest_point)
            print(f"[FakeInflux] New data point added. Buffer size: {len(self.buffer)}")
            
            if self.point_counter >= 5:
                data = list(self.buffer)
                mean_values = self.stats_service.compute_hourly_mean(data)
                if mean_values and self.collection:
                    try:
                        # Insert mean into MongoDB
                        self.collection.insert_one(mean_values)
                        print(f"[FakeInflux] Mean inserted into MongoDB. Document: {mean_values}")
                        
                        # Retrieve last 3 means immediately after insertion
                        self.last_lookback = list(self.collection.find().sort("_id", -1).limit(3))
                        print(f"[FakeInflux] Retrieved last 3 means: {len(self.last_lookback)} items")
                        
                        # Clear buffer for next non-overlapping batch
                        self.buffer.clear()
                        print("[FakeInflux] Buffer cleared for next batch.")
                        
                    except Exception as e:
                        print(f"[FakeInflux] Error inserting or retrieving from MongoDB: {e}")
                self.point_counter = 0

            time.sleep(self.interval)

    def get_last_360_points(self):
        """Return the rolling window (up to 360 points)."""
        return list(self.buffer)
    
    def get_recent_means_from_db(self, limit=10):
        """Optional: Retrieve recent means from MongoDB for API or testing."""
        if self.collection:
            return list(self.collection.find().sort("_id", -1).limit(limit))
        return []
    
    def get_last_lookback(self):
        """Return the last 3 retrieved means."""
        return self.last_lookback


