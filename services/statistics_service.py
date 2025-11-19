# services/statistics_service.py

import numpy as np

class StatisticsService:

    @staticmethod
    # def compute_hourly_mean(data):
    #     """
    #     Compute mean for each feature from 360 data points.
    #     Expects a list of dictionaries.
    #     """
    #     if len(data) == 0:
    #         return None

    #     temp_shaft = np.mean([d["temp_shaft"] for d in data])
    #     temp_body = np.mean([d["temp_body"] for d in data])
    #     current = np.mean([d["current"] for d in data])
    #     vibration = np.mean([d["vibration"] for d in data])

    #     return {
    #         "temp_shaft_mean": round(float(temp_shaft), 3),
    #         "temp_body_mean": round(float(temp_body), 3),
    #         "current_mean": round(float(current), 3),
    #         "vibration_mean": round(float(vibration), 3),
    #         "num_points_used": len(data)
    #     }
    def compute_hourly_mean(data_points):
        if not data_points:
            return None

        # Extract features
        temp_body_values = [p['temp_body'] for p in data_points]
        temp_shaft_values = [p['temp_shaft'] for p in data_points]
        vibration_x_values = [p['vibration_x'] for p in data_points]
        vibration_y_values = [p['vibration_y'] for p in data_points]
        vibration_z_values = [p['vibration_z'] for p in data_points]
        current_values = [p['current'] for p in data_points]

        # Calculate RMS for vibration
        vibration_values = []
        for i in range(len(data_points)):
            rms = (vibration_x_values[i]**2 + vibration_y_values[i]**2 + vibration_z_values[i]**2)**0.5  # New: RMS calculation
            vibration_values.append(rms)

        # Compute means
        mean_values = {
            "temp_body_mean": round(np.mean(temp_body_values), 3),
            "temp_shaft_mean": round(np.mean(temp_shaft_values), 3),
            "vibration_mean": round(np.mean(vibration_values), 3),
            "current_mean": round(np.mean(current_values), 3),
            "num_points_used": len(data_points)
        }
        return mean_values




