import time
from collections import deque
from influxdb_client import InfluxDBClient
from services.inference_service_2 import InferenceService
from configs.mongodb_config import influx_url, influx_token, influx_org, influx_bucket, workspace_id

class RealtimeInfluxStreamer:
    def __init__(self, interval_seconds=10, max_points=160):
        self.influx_client = InfluxDBClient(
            url=influx_url, 
            token=influx_token, 
            org=influx_org
        )
        self.influx_bucket = influx_bucket

        self.latest_point = None
        self.buffer = deque(maxlen=max_points)
        self.interval = interval_seconds
        self.last_time = None
        self.point_counter = 0
        self.last_lookback = []

    def start_stream(self):
        """
        Continuously poll InfluxDB for new data every 10 seconds.
        Buffer maintains the last 160 data points.
        Every 160 points, prepare lookback and run inference.
        """
        while True:
            # Query InfluxDB for the latest data point timestamp
            timestamp_query = f'''
            from(bucket: "{self.influx_bucket}")
              |> range(start: -1m)
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

                if latest_time and latest_time != self.last_time:
                    # Fetch full data for the latest timestamp
                    full_query = f'''
                    from(bucket: "{self.influx_bucket}")
                      |> range(start: -1m)
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
                            if record["_time"] == latest_time:
                                new_point = {
                                    "timestamp": record.get_time().isoformat(),
                                    "temp_body": float(record.values.get("temp_body", 0)),
                                    "temp_shaft": float(record.values.get("temp_shaft", 0)),
                                    "vibration_x": float(record.values.get("vibration_x", 0)),
                                    "vibration_y": float(record.values.get("vibration_y", 0)),
                                    "vibration_z": float(record.values.get("vibration_z", 0)),
                                    "current": float(record.values.get("current", 0)),
                                    "machine_id": workspace_id,
                                }
                                self.latest_point = new_point
                                self.buffer.append(new_point)
                                self.point_counter += 1
                                print("[RealtimeInflux] New data point from InfluxDB:", self.latest_point)
                                print(f"[RealtimeInflux] Buffer size: {len(self.buffer)}/160")
                                break
                            
                    self.last_time = latest_time
                else:
                    print("[RealtimeInflux] No new data from InfluxDB.")
            except Exception as e:
                print(f"[RealtimeInflux] Error querying InfluxDB: {e}")

            # When buffer is full (160 points), prepare lookback and run inference
            if self.point_counter >= 160:
                data = list(self.buffer)
                try:
                    # Store the current buffer as lookback for model input
                    self.last_lookback = data.copy()
                    print(f"[RealtimeInflux] Lookback prepared: {len(self.last_lookback)} data points")

                    # Run inference with the 160 data points
                    inference = InferenceService()
                    forecast, alerts = inference.run_inference(self.last_lookback)
                    print(f"[RealtimeInflux] Forecast shape: {forecast.shape if forecast is not None else 'None'}")
                    print(f"[RealtimeInflux] Alerts: {alerts}")

                except Exception as e:
                    print(f"[RealtimeInflux] Error in inference: {e}")
                
                # Reset counter (buffer will continue rolling)
                self.point_counter = 0

            time.sleep(self.interval)

    def get_buffer(self):
        """Return the current buffer (up to 160 points)."""
        return list(self.buffer)
    
    def get_latest_point(self):
        """Return the most recent data point."""
        return self.latest_point
    
    def get_last_lookback(self):
        """Return the last 160 data points used for inference."""
        return self.last_lookback