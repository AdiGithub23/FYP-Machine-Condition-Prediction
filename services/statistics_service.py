# services/statistics_service.py

import numpy as np

class StatisticsService:

    @staticmethod
    def compute_hourly_mean(data):
        """
        Compute mean for each feature from 360 data points.
        Expects a list of dictionaries.
        """
        if len(data) == 0:
            return None

        temp_shaft = np.mean([d["temp_shaft"] for d in data])
        temp_body = np.mean([d["temp_body"] for d in data])
        current = np.mean([d["current"] for d in data])
        vibration = np.mean([d["vibration"] for d in data])

        return {
            "temp_shaft_mean": round(float(temp_shaft), 3),
            "temp_body_mean": round(float(temp_body), 3),
            "current_mean": round(float(current), 3),
            "vibration_mean": round(float(vibration), 3),
            "num_points_used": len(data)
        }
