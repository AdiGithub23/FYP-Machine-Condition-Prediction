# services/fake_influx_streamer.py

import time
from collections import deque
from influxdb_client import InfluxDBClient
from services.statistics_service import StatisticsService
from services.inference_service import InferenceService
from configs.mongodb_config import get_database
from configs.mongodb_config import influx_url, influx_token, influx_org, influx_bucket, workspace_id

class RealInfluxStreamer:
    def __init__(self, interval_seconds=10, max_points=360):
        self.influx_client = InfluxDBClient(
            url=influx_url, 
            token=influx_token, 
            org=influx_org
        )
        self.influx_bucket = influx_bucket

        self.latest_point = None
        self.buffer = deque(maxlen=max_points)
        self.interval = interval_seconds
        
        self.stats_service = StatisticsService()
        self.point_counter = 0
        self.db = get_database()
        self.collection = self.db[f"hourly_means_{workspace_id}"] if self.db else None
        self.last_lookback = []

    def start_stream(self):
        """
        Continuously poll InfluxDB for new data every 10 seconds.
        Every 360 points, compute mean, insert to MongoDB, retrieve lookback, run inference.
        """
        while True:
            # Query InfluxDB for the latest data point (last 10 seconds)
            timestamp_query = f'''
            from(bucket: "{self.influx_bucket}")
              |> range(start: -2m)
              |> filter(fn: (r) => r._measurement == "machine_metrics")
              |> filter(fn: (r) => r.machine_id == "{workspace_id}")
              |> keep(columns: ["_time"])
              |> sort(columns: ["_time"], desc: true)
              |> limit(n: 1)
            '''

            try:
                timestamp_result = self.influx_client.query_api().query(timestamp_query)
                latest_time = None
                for table in timestamp_result:
                    for record in table.records:
                        latest_time = record.get_time()

                if latest_time and latest_time != getattr(self, 'last_time', None):
                    # Step 2: Fetch full data for the latest timestamp
                    full_query = f'''
                    from(bucket: "{self.influx_bucket}")
                      |> range(start: -5m)
                      |> filter(fn: (r) => r["_measurement"] == "machine_metrics")
                      |> filter(fn: (r) => r["machine_id"] == "{workspace_id}")
                      |> pivot(
                          rowKey: ["_time"],
                          columnKey: ["_field"],
                          valueColumn: "_value"
                      )
                      |> sort(columns: ["_time"], desc: false)
                    '''

                    full_data = self.influx_client.query_api().query(full_query)

                    for table in full_data:
                        for record in table.records:
                            if record["_time"] == latest_time:  # Match the latest timestamp
                                new_point = {
                                    "timestamp": record.get_time().isoformat(),
                                    "temp_body": float(record.values.get("temp_body", 0)),
                                    "temp_shaft": float(record.values.get("temp_shaft", 0)),
                                    "vibration_x": float(record.values.get("vibration_x", 0)),
                                    "vibration_y": float(record.values.get("vibration_y", 0)),
                                    "vibration_z": float(record.values.get("vibration_z", 0)),
                                    "current": float(record.values.get("current", 0)),
                                    "machine_id": workspace_id,  # Use workspace_id
                                }
                                self.latest_point = new_point
                                self.buffer.append(new_point)
                                self.point_counter += 1
                                print("[RealInflux] New data point from InfluxDB:", self.latest_point)
                                print(f"[RealInflux] Buffer size: {len(self.buffer)}")
                                break  # Only process the matching record
                            
                    self.last_time = latest_time  # Track last processed time
                else:
                    print("[RealInflux] No new data from InfluxDB.")
            except Exception as e:
                print(f"[RealInflux] Error querying InfluxDB: {e}")

            if self.point_counter >= 360:
                data = list(self.buffer)
                mean_values = self.stats_service.compute_hourly_mean(data)
                if mean_values and self.collection:
                    try:
                        mean_values["machine_id"] = data[0]["machine_id"] if data else "unknown"
                        self.collection.insert_one(mean_values)
                        print(f"[RealInflux] Mean inserted into MongoDB: {mean_values}")

                        self.last_lookback = list(self.collection.find({"machine_id": workspace_id}).sort("_id", -1).limit(1200))
                        print(f"[RealInflux] Retrieved lookback: {len(self.last_lookback)} items")

                        inference = InferenceService()
                        forecast, alerts = inference.run_inference(self.last_lookback)
                        print(f"[RealInflux] Forecast shape: {forecast.shape if forecast is not None else 'None'}")
                        print(f"[RealInflux] Alerts: {alerts}")

                        self.buffer.clear()
                        print("[RealInflux] Buffer cleared.")

                    except Exception as e:
                        print(f"[RealInflux] Error in MongoDB/inference: {e}")
                self.point_counter = 0

            time.sleep(self.interval)

    def get_last_360_points(self):
        """Return the rolling window (up to 360 points)."""
        return list(self.buffer)
    
    def get_recent_means_from_db(self, limit=360):
        """Optional: Retrieve recent means from MongoDB for API or testing."""
        if self.collection:
            return list(self.collection.find().sort("_id", -1).limit(limit))
        return []
    
    def get_last_lookback(self):
        """Return the last 1200 retrieved means."""
        return self.last_lookback


