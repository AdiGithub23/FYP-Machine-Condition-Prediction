# python main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from services.fake_influx_reader import FakeInfluxReader
# from services.fake_influx_streamer import FakeInfluxStreamer
# from services.real_influx_streamer import RealInfluxStreamer
from services.real_influx_streamer_2 import RealtimeInfluxStreamer
from services.real_influx_streamer_3 import ScheduledInfluxInference
from services.statistics_service import StatisticsService
import threading

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# fake_influx = FakeInfluxReader()
# streamer = FakeInfluxStreamer(interval_seconds=10, max_points=360)
# streamer = RealInfluxStreamer(interval_seconds=10, max_points=360)
# streamer = RealtimeInfluxStreamer(interval_seconds=10, max_points=160)
streamer = ScheduledInfluxInference(inference_interval_seconds=180)  # 3 minutes

stats_service = StatisticsService()


@app.get("/sensor/means-history-db")
def get_means_history_from_db():
    """
    Return recent mean values from MongoDB.
    """
    data = streamer.get_recent_means_from_db()
    return {
        "status": "success",
        "means_collected": len(data),
        "data": data
    }

@app.on_event("startup")
# def start_background_stream():
#     """
#     Start the 10-second fake data generator in the background.
#     """
#     thread = threading.Thread(target=streamer.start_stream, daemon=True)
#     thread.start()
#     print("Background data streaming process started.")
def start_background_inference():
    """
    Start the scheduled inference process in the background.
    Runs inference every 3 minutes automatically.
    """
    thread = threading.Thread(target=streamer.start_stream, daemon=True)
    thread.start()
    print("Background inference scheduler started (runs every 3 minutes).")

# API endpoints to monitor the process
@app.get("/inference/last-prediction")
def get_last_prediction():
    """
    Get the most recent prediction results including forecast and alerts.
    """
    prediction_data = streamer.get_last_prediction()
    
    if prediction_data is None:
        return {
            "status": "error",
            "message": "No predictions available yet. Wait for first inference cycle.",
            "forecast": None,
            "alerts": None
        }
    
    # Generate future timestamps for forecast (10 second intervals)
    import datetime
    last_lookback = streamer.get_last_lookback()
    if last_lookback and len(last_lookback) > 0:
        last_timestamp = datetime.datetime.fromisoformat(last_lookback[-1]['timestamp'].replace('Z', '+00:00'))
        future_timestamps = [
            (last_timestamp + datetime.timedelta(seconds=10 * (i + 1))).isoformat()
            for i in range(20)
        ]
    else:
        future_timestamps = None
    
    # Format forecast output with timestamps
    forecast_formatted = streamer.inference_service.format_forecast_output(
        prediction_data["forecast"], 
        timestamps=future_timestamps
    )
    
    return {
        "status": "success",
        "message": "Last prediction retrieved successfully",
        "inference_count": prediction_data["inference_count"],
        "timestamp": prediction_data["timestamp"],
        "alerts": prediction_data["alerts"],
        "forecast": forecast_formatted,
        "forecast_array": prediction_data["forecast"].tolist()  # Raw numpy array as list
    }
    
@app.get("/inference/status")
def get_inference_status():
    """
    Get the status of the inference scheduler including timing information.
    """
    status_data = streamer.get_inference_status()
    return {
        "status": "success",
        "data": status_data
    }


@app.get("/sensor/latest")
def get_latest_sensor_point():
    """
    Return the most recent datapoint.
    """
    # return {
    #     "status": "success",
    #     "msg": "Latest data retrieved successfully",
    #     "data": streamer.latest_point
    # }

    # For REXTRO Demo
    data = streamer.get_latest_point()
    if data is None:
        return {
            "status": "error",
            "msg": "No data available yet. Wait for first inference cycle.",
            "data": None
        }
    return {
        "status": "success",
        "msg": "Latest data retrieved successfully",
        "data": data
    }
# For REXTRO Demo
@app.get("/sensor/buffer")
def get_buffer_data():
    """
    Return all data points in the buffer (last 160 points).
    """
    data = streamer.get_buffer()
    return {
        "status": "success",
        "points_collected": len(data),
        "data": data
    }


@app.get("/sensor/history")
def get_last_hour_points():
    data = streamer.get_last_360_points()
    return {
        "status": "success",
        "points_collected": len(data),
        "data": data
    }

@app.get("/sensor/hourly-mean")
def get_hourly_mean():
    data = streamer.get_last_360_points()
    mean_values = stats_service.compute_hourly_mean(data)

    # if mean_values is None:
    if mean_values is None or len(data) < 360:
        return {
            "status": "error",
            "msg": "Not enough data points yet (need at least 1)."
        }

    return {
        "status": "success",
        "hourly_mean": mean_values
    }

@app.get("/sensor/means-history")
def get_means_history():
    """
    Return the list of automatically calculated mean values.
    """
    return {
        "status": "success",
        "means_collected": len(streamer.get_means_list()),
        "data": streamer.get_means_list()
    }

@app.get("/sensor/last-lookback")
def get_last_lookback():
    """
    Retrieve the last 1200 mean values from MongoDB (updated every 1 hour).
    """
    data = streamer.get_last_lookback()
    return {
        "status": "success",
        "means_collected": len(data),
        "data": data
    }





