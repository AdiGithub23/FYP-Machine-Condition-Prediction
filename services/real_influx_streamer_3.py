# services/real_influx_streamer_3.py
import time
from datetime import datetime
from influxdb_client import InfluxDBClient
from services.inference_service_2 import InferenceService
from configs.mongodb_config import influx_url, influx_token, influx_org, influx_bucket, workspace_id

class ScheduledInfluxInference:
    def __init__(self, inference_interval_seconds=180):
        """
        Initialize scheduled inference service.
        Runs inference every N seconds (default: 180 = 3 minutes).
        
        Args:
            inference_interval_seconds (int): Time between inference runs (default: 180 seconds = 3 minutes)
        """
        self.influx_client = InfluxDBClient(
            url=influx_url, 
            token=influx_token, 
            org=influx_org
        )
        self.influx_bucket = influx_bucket
        self.workspace_id = workspace_id
        
        self.inference_interval = inference_interval_seconds
        self.last_prediction = None
        self.last_alerts = None
        self.last_lookback = []
        self.next_inference_time = None
        self.inference_count = 0
        
        # Initialize inference service
        print(f"[ScheduledInflux] Initializing inference service...")
        self.inference_service = InferenceService()
        print(f"[ScheduledInflux] Inference will run every {inference_interval_seconds} seconds ({inference_interval_seconds/60:.1f} minutes)")

    def start_stream(self):
        """
        Continuously run inference every N seconds.
        
        Workflow:
        1. Wait for inference interval (3 minutes)
        2. Query InfluxDB for last 160 data points
        3. Run inference to predict next 20 points
        4. Check for anomalies and log alerts
        5. Repeat
        """
        print("\n" + "="*80)
        print("[ScheduledInflux] Starting Scheduled Inference Loop")
        print("="*80)
        
        while True:
            try:
                # Calculate next inference time
                self.next_inference_time = datetime.now()
                next_time_str = self.next_inference_time.strftime("%Y-%m-%d %H:%M:%S")
                
                print(f"\n[ScheduledInflux] Inference #{self.inference_count + 1} starting at {next_time_str}")
                print("-" * 80)
                
                # Query last 160 data points from InfluxDB
                print(f"[ScheduledInflux] Querying last 160 data points from InfluxDB...")
                buffer_data = self.query_last_160_points()
                
                if buffer_data is None or len(buffer_data) < 160:
                    print(f"⚠️ [ScheduledInflux] Insufficient data: {len(buffer_data) if buffer_data else 0}/160 points")
                    print(f"[ScheduledInflux] Waiting {self.inference_interval} seconds before retry...")
                    time.sleep(self.inference_interval)
                    continue
                
                print(f"✅ [ScheduledInflux] Retrieved {len(buffer_data)} data points")
                
                # Store lookback for API access
                self.last_lookback = buffer_data.copy()
                
                # Run inference
                print(f"[ScheduledInflux] Running inference...")
                forecast, alerts = self.inference_service.run_inference(buffer_data)
                
                if forecast is not None:
                    # Store prediction results
                    self.last_prediction = forecast
                    self.last_alerts = alerts
                    self.inference_count += 1
                    
                    print(f"✅ [ScheduledInflux] Inference completed successfully")
                    print(f"   Forecast shape: {forecast.shape}")
                    print(f"   Alert status: {alerts['status']}")
                    print(f"   Alert message: {alerts['message']}")
                else:
                    print(f"❌ [ScheduledInflux] Inference failed: {alerts['message']}")
                
                # Wait for next inference cycle
                print(f"\n[ScheduledInflux] Next inference in {self.inference_interval} seconds ({self.inference_interval/60:.1f} minutes)...")
                print("-" * 80)
                time.sleep(self.inference_interval)
                
            except Exception as e:
                print(f"❌ [ScheduledInflux] Error in inference loop: {e}")
                import traceback
                traceback.print_exc()
                print(f"[ScheduledInflux] Retrying in {self.inference_interval} seconds...")
                time.sleep(self.inference_interval)

    def query_last_160_points(self):
        """
        Query InfluxDB for the last 160 data points.
        This covers approximately the last 26.67 minutes of data (160 points × 10 seconds).
        
        Returns:
            list: List of 160 dictionaries with sensor data, or None if error
        """
        # Query range: 30 minutes to ensure we get at least 160 points
        query = f'''
        from(bucket: "{self.influx_bucket}")
          |> range(start: -30m)
          |> filter(fn: (r) => r["_measurement"] == "machine_metrics")
          |> filter(fn: (r) => r["machine_id"] == "{self.workspace_id}")
          |> pivot(
              rowKey: ["_time"],
              columnKey: ["_field"],
              valueColumn: "_value"
          )
          |> sort(columns: ["_time"], desc: false)
          |> tail(n: 160)
        '''
        
        try:
            result = self.influx_client.query_api().query(query)
            data_points = []
            
            for table in result:
                for record in table.records:
                    data_point = {
                        "timestamp": record.get_time().isoformat(),
                        "temp_body": float(record.values.get("temp_body", 0)),
                        "temp_shaft": float(record.values.get("temp_shaft", 0)),
                        "vibration_x": float(record.values.get("vibration_x", 0)),
                        "vibration_y": float(record.values.get("vibration_y", 0)),
                        "vibration_z": float(record.values.get("vibration_z", 0)),
                        "current": float(record.values.get("current", 0)),
                        "machine_id": self.workspace_id,
                    }
                    data_points.append(data_point)
            
            print(f"[ScheduledInflux] Query returned {len(data_points)} data points")
            
            if len(data_points) > 0:
                first_time = data_points[0]['timestamp']
                last_time = data_points[-1]['timestamp']
                print(f"[ScheduledInflux] Time range: {first_time} to {last_time}")
            
            return data_points
            
        except Exception as e:
            print(f"❌ [ScheduledInflux] Error querying InfluxDB: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_last_prediction(self):
        """
        Get the most recent prediction results.
        
        Returns:
            dict: Contains forecast array and alerts, or None if no prediction yet
        """
        if self.last_prediction is None:
            return None
        
        return {
            "forecast": self.last_prediction,
            "alerts": self.last_alerts,
            "inference_count": self.inference_count,
            "timestamp": self.last_alerts.get("timestamp") if self.last_alerts else None
        }
    
    def get_inference_status(self):
        """
        Get the status of the inference scheduler.
        
        Returns:
            dict: Status information including next run time and inference count
        """
        return {
            "status": "running",
            "inference_interval_seconds": self.inference_interval,
            "inference_interval_minutes": self.inference_interval / 60,
            "total_inferences_run": self.inference_count,
            "last_inference_time": self.last_alerts.get("timestamp") if self.last_alerts else None,
            "next_inference_time": self.next_inference_time.isoformat() if self.next_inference_time else None,
            "has_prediction": self.last_prediction is not None
        }
    
    def get_last_lookback(self):
        """
        Return the last 160 data points used for inference.
        
        Returns:
            list: Last lookback data (160 points)
        """
        return self.last_lookback
    
    def get_buffer(self):
        """
        Return the current lookback buffer (for compatibility with API endpoints).
        Same as get_last_lookback().
        
        Returns:
            list: Last lookback data (160 points)
        """
        return self.last_lookback
    
    def get_latest_point(self):
        """
        Return the most recent data point from the last lookback.
        
        Returns:
            dict: Most recent data point, or None if no data yet
        """
        if self.last_lookback and len(self.last_lookback) > 0:
            return self.last_lookback[-1]
        return None
