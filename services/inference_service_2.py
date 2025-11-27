import numpy as np
import pandas as pd
import torch
import pickle
import os
from datetime import datetime

class InferenceService:
    def __init__(self):
        """
        Initialize the inference service with model and scaler.
        Matches the configuration from Demo_PatchTST_mini_1.ipynb
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(current_dir, "..", "AI-Model-Artifacts", "V2", "artifacts")
        base_dir = os.path.abspath(base_dir)
        
        self.model_path = os.path.join(base_dir, "patchtst_full_model.pt")
        self.scaler_path = os.path.join(base_dir, "scaler.pkl")        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.context_length = 160
        self.prediction_length = 20
        self.num_features = 6
        
        self.feature_names = [
            'temp_body', 'temp_shaft', 'vibration_x', 
            'vibration_y', 'vibration_z', 'current'
        ]
        
        # Load scaler
        print(f"[InferenceService] Loading scaler from: {self.scaler_path}")
        try:
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print("✅ Scaler loaded successfully")
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
            raise
        
        # Load model
        print(f"[InferenceService] Loading model from: {self.model_path}")
        try:
            self.model = torch.load(
                self.model_path, 
                map_location=self.device, 
                weights_only=False
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def run_inference(self, buffer_data):
        """
        Run inference on 160 data points to predict next 20 points.
        
        Args:
            buffer_data (list of dicts): 160 data points from RealtimeInfluxStreamer
                Each dict contains: timestamp, temp_body, temp_shaft, vibration_x, 
                                   vibration_y, vibration_z, current, machine_id
        
        Returns:
            forecast (np.ndarray): Predicted 20 points, shape (20, 6)
            alerts (dict): Status and message
        """
        print("\n" + "="*80)
        print("[InferenceService] Starting Inference")
        print("="*80)
        
        # Validate input
        if len(buffer_data) != self.context_length:
            error_msg = f"Expected {self.context_length} data points, got {len(buffer_data)}"
            print(f"❌ {error_msg}")
            return None, {"status": "error", "message": error_msg}
        
        try:
            # Step 1: Extract features from buffer data
            print(f"\n[Step 1] Extracting features from {len(buffer_data)} data points...")
            raw_data = pd.DataFrame([
                [
                    point['temp_body'],
                    point['temp_shaft'],
                    point['vibration_x'],
                    point['vibration_y'],
                    point['vibration_z'],
                    point['current']
                ]
                for point in buffer_data
            ], columns=self.feature_names)
            
            print(f"✅ Raw data extracted, shape: {raw_data.shape}")
            # print(f"   First 5 rows:\n{raw_data.head()}")
            
            # Step 2: Scale the data
            print(f"\n[Step 2] Scaling data using loaded scaler...")
            scaled_data = self.scaler.transform(raw_data.values)
            print(f"✅ Data scaled, shape: {scaled_data.shape}")
            print(f"   Scaled mean: {scaled_data.mean():.6f}, std: {scaled_data.std():.6f}")
            
            # Step 3: Prepare input tensor for model
            print(f"\n[Step 3] Preparing input tensor for model...")
            # Reshape: (160, 6) -> (1, 160, 6) for batch processing
            model_input = scaled_data.reshape(1, self.context_length, self.num_features)
            input_tensor = torch.tensor(model_input, dtype=torch.float32).to(self.device)
            print(f"✅ Input tensor prepared, shape: {input_tensor.shape}")
            
            # Step 4: Run model inference
            print(f"\n[Step 4] Running model inference...")
            with torch.no_grad():
                outputs = self.model(past_values=input_tensor)
                preds = outputs.prediction_outputs
                
                # Handle tuple output (if model returns tuple)
                if isinstance(preds, tuple):
                    preds = preds[0]
                
                # Convert to numpy: (1, 20, 6) -> (20, 6)
                forecast_scaled = preds.squeeze(0).cpu().numpy()
            
            print(f"✅ Model inference complete, forecast shape: {forecast_scaled.shape}")
            
            # Step 5: Inverse transform to original scale
            print(f"\n[Step 5] Inverse transforming predictions to original scale...")
            forecast_original = self.scaler.inverse_transform(forecast_scaled)
            print(f"✅ Forecast in original scale, shape: {forecast_original.shape}")
            
            # Step 6: Prepare output
            print(f"\n[Step 6] Preparing output...")
            forecast_df = pd.DataFrame(forecast_original, columns=self.feature_names)
            print(f"✅ Forecast DataFrame:\n{forecast_df}")
            
            # Step 7: Anomaly detection (optional)
            alerts = self._detect_anomalies(scaled_data, forecast_scaled)
            
            print("\n" + "="*80)
            print("[InferenceService] Inference Complete")
            print("="*80)
            
            return forecast_original, alerts
        
        except Exception as e:
            error_msg = f"Error during inference: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return None, {"status": "error", "message": error_msg}

    def _detect_anomalies(self, lookback_scaled, forecast_scaled):
        """
        Detect anomalies in the forecast by comparing with lookback statistics.
        
        Args:
            lookback_scaled (np.ndarray): Scaled lookback data (160, 6)
            forecast_scaled (np.ndarray): Scaled forecast data (20, 6)
        
        Returns:
            dict: Alert status and message
        """
        print(f"\n[Anomaly Detection] Analyzing forecast for anomalies...")
        
        at_risk_features = []
        
        for feature_idx, feature_name in enumerate(self.feature_names):
            # Get lookback statistics
            lookback_feature = lookback_scaled[:, feature_idx]
            max_lookback = np.max(lookback_feature)
            min_lookback = np.min(lookback_feature)
            mean_lookback = np.mean(lookback_feature)
            std_lookback = np.std(lookback_feature)
            
            # Get forecast values
            forecast_feature = forecast_scaled[:, feature_idx]
            
            # Count anomalies (values outside lookback range)
            num_exceeding_max = np.sum(forecast_feature > max_lookback)
            num_below_min = np.sum(forecast_feature < min_lookback)
            total_anomalous = num_exceeding_max + num_below_min
            anomaly_percentage = (total_anomalous / self.prediction_length) * 100
            
            print(f"   {feature_name}: {anomaly_percentage:.1f}% anomalies "
                  f"({num_exceeding_max} above max, {num_below_min} below min)")
            
            # Flag feature if anomaly percentage exceeds threshold
            if anomaly_percentage >= 30:  # 30% threshold
                at_risk_features.append(feature_name)
        
        # Generate alert message
        current_time = datetime.now().isoformat()
        if at_risk_features:
            alert_message = (f"⚠️ Machine at Risk: Anomalies detected in "
                           f"{', '.join(at_risk_features)} (Checked at: {current_time})")
            status = "warning"
        else:
            alert_message = f"✅ Machine Condition Normal (Checked at: {current_time})"
            status = "success"
        
        print(f"\n{alert_message}")
        
        return {
            "status": status,
            "message": alert_message,
            "at_risk_features": at_risk_features,
            "timestamp": current_time
        }

    def format_forecast_output(self, forecast, timestamps=None):
        """
        Format forecast output as a structured dictionary.
        
        Args:
            forecast (np.ndarray): Forecast array (20, 6)
            timestamps (list, optional): List of timestamp strings
        
        Returns:
            dict: Formatted forecast data
        """
        forecast_list = []
        for i in range(self.prediction_length):
            point = {
                "step": i + 1,
                "temp_body": float(forecast[i, 0]),
                "temp_shaft": float(forecast[i, 1]),
                "vibration_x": float(forecast[i, 2]),
                "vibration_y": float(forecast[i, 3]),
                "vibration_z": float(forecast[i, 4]),
                "current": float(forecast[i, 5])
            }
            if timestamps and i < len(timestamps):
                point["timestamp"] = timestamps[i]
            forecast_list.append(point)
        
        return {
            "forecast_horizon": self.prediction_length,
            "num_features": self.num_features,
            "predictions": forecast_list
        }