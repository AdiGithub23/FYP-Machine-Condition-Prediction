# services/fake_influx_streamer.py

import time
from collections import deque
from fake_data.data_generator import generate_fake_point
from services.statistics_service import StatisticsService

class FakeInfluxStreamer:
    def __init__(self, interval_seconds=10, max_points=360):
        self.latest_point = None
        self.buffer = deque(maxlen=max_points)
        self.interval = interval_seconds
        
        self.stats_service = StatisticsService()
        self.means_list = []
        self.point_counter = 0

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
                if mean_values:
                    self.means_list.append(mean_values)
                    print(f"[FakeInflux] Mean calculated and added to list. List size: {len(self.means_list)}")
                self.point_counter = 0

            time.sleep(self.interval)

    def get_last_360_points(self):
        """Return the rolling window (up to 360 points)."""
        return list(self.buffer)
    
    def get_means_list(self):
        """Return the list of calculated means (for DB storage or API)."""
        return self.means_list



