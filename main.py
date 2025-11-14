# python main.py

from fastapi import FastAPI
from services.fake_influx_reader import FakeInfluxReader
from services.fake_influx_streamer import FakeInfluxStreamer
from services.statistics_service import StatisticsService
import threading

app = FastAPI()
# fake_influx = FakeInfluxReader()
streamer = FakeInfluxStreamer(interval_seconds=3, max_points=5)
stats_service = StatisticsService()


@app.on_event("startup")
def start_background_stream():
    """
    Start the 10-second fake data generator in the background.
    """
    thread = threading.Thread(target=streamer.start_stream, daemon=True)
    thread.start()
    print("Background fake data generator started.")

@app.get("/sensor/latest")
def get_latest_sensor_point():
    """
    Return the most recent generated datapoint.
    """
    return {
        "status": "success",
        "msg": "Latest data retrieved successfully",
        "data": streamer.latest_point
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
    if mean_values is None or len(data) < 5:
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
