# fake_data/data_generator.py

import time
import random
from datetime import datetime, timedelta

def generate_fake_point():
    """
    Generate one fake sensor data point.
    Simulates real-time data coming every 10 seconds.
    """

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "temp_shaft": round(random.uniform(25.0, 45.0), 3),
        "temp_body": round(random.uniform(25.0, 45.0), 3),
        "current": round(random.uniform(4.0, 7.0), 3),
        "vibration": round(random.uniform(7.5, 9.5), 3),
    }


def generate_fake_timeseries(n_points: int = 360):
    """
    Generate n_points spaced 10 seconds apart.
    Used until real InfluxDB is connected.
    """

    now = datetime.utcnow()
    data = []

    for i in range(n_points):
        timestamp = now - timedelta(seconds=10 * i)
        # timestamp = now - timedelta(seconds=1 * i)
        entry = generate_fake_point()
        entry["timestamp"] = timestamp.isoformat()
        data.append(entry)

    return list(reversed(data))


