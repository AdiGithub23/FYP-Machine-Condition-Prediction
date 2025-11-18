# services/inference_service.py

import numpy as np
import pandas as pd
import torch
import pickle
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from configs.mongodb_config import get_database

class InferenceService:
    def __init__(self):
        # Define paths and device
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(current_dir, "..", "AI-Model-Artifacts")
        base_dir = os.path.abspath(base_dir)
        # base_dir = "../AI-Model-Artifacts/"
        self.scaler_path = os.path.join(base_dir, "MinMax_Scaler.pkl")
        self.model_path = os.path.join(base_dir, "multi_vanilla_patchtst.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load scaler and model
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.model = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.to(self.device)
        self.model.eval()
        
        # MongoDB connection
        self.db = get_database()
        self.collection = self.db["hourly_means"] if self.db else None
        self.users_collection = self.db["users"] if self.db else None
        self.workspaces_collection = self.db["workspaces"] if self.db else None

    def run_inference(self, retrieved_docs):
        """
        Run the full inference pipeline: preprocess data, feed to model, detect anomalies, update DB, send email.
        Args: retrieved_docs (list of dicts from MongoDB, last 1200).
        Returns: forecast (np.array), alerts (dict with status and message).
        """
        if len(retrieved_docs) < 1200:
            print("[Inference] Not enough data (need 1200). Skipping inference.")
            return None, {"status": "error", "message": "Not enough data for inference."}
        
        # Step 1: Extract features (from notebook Cell 3)
        raw_data = pd.DataFrame([
            [doc['temp_body_mean'], doc['temp_shaft_mean'], doc['current_mean'], doc['vibration_mean']]
            for doc in retrieved_docs
        ], columns=['temp_body', 'temp_shaft', 'current', 'vibration_magnitude'])
        
        # Step 2: Scale data (from notebook Cell 5)
        scaled_data = self.scaler.transform(raw_data)
        scaled_data = np.clip(scaled_data, 0, 1)
        
        # Step 3: Reshape for model (from notebook Cell 6)
        model_input = scaled_data.reshape(1, 1200, 4)
        input_tensor = torch.tensor(model_input, dtype=torch.float32).to(self.device)
        
        # Step 4: Feed to model and get forecast (from notebook Cell 7)
        with torch.no_grad():
            outputs = self.model(past_values=input_tensor)
            forecast = outputs.prediction_outputs.squeeze().cpu().numpy()
        
        # Step 5: Anomaly detection (from notebook Cell 9)
        num_features = 4
        horizon = 240
        feature_names = ['temp_body', 'temp_shaft', 'current', 'vibration_magnitude']
        at_risk_features = []
        
        for feature_idx in range(num_features):
            forecast_feature = forecast[:, feature_idx]
            max_lookback = np.max(scaled_data[:, feature_idx])
            min_lookback = np.min(scaled_data[:, feature_idx])
            num_exceeding_max = np.sum(forecast_feature > max_lookback)
            num_below_min = np.sum(forecast_feature < min_lookback)
            total_anomalous = num_exceeding_max + num_below_min
            anomaly_percentage = (total_anomalous / horizon) * 100
            
            print(f"[Inference] Feature {feature_idx}: Anomaly % = {anomaly_percentage:.2f}%")
            
            if anomaly_percentage >= 30:
                at_risk_features.append(feature_names[feature_idx])
        
        # Determine alert message
        if at_risk_features:
            overall_at_risk = True
            alert_message = f"Machine at Risk: Stay alert on {', '.join(at_risk_features)}"
        else:
            overall_at_risk = False
            alert_message = "Machine Condition Normal"
        
        # Step 6: Update MongoDB with alert (from notebook Cell 10)
        if self.collection:
            latest_doc = self.collection.find_one(sort=[("_id", -1)])
            if latest_doc:
                self.collection.update_one(
                    {"_id": latest_doc["_id"]},
                    {"$set": {"alert_message": alert_message}}
                )
                print(f"[Inference] Alert message '{alert_message}' inserted into latest document.")
            else:
                print("[Inference] No documents found in 'hourly_means' collection.")
        else:
            print("[Inference] Database connection failed.")
        
        # Step 7: Send email if at risk (from notebook Cell 10)
        # if overall_at_risk:
        #     to_email = "aadhiganegoda@gmail.com"
        #     from_email = "thisupun3@gmail.com"
        #     subject = "Machine Condition Alert"
        #     body = alert_message
            
        #     sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
        #     if not sendgrid_api_key:
        #         print("[Inference] SendGrid API Key not found.")
        #     else:
        #         message = Mail(from_email=from_email, to_emails=to_email, subject=subject, plain_text_content=body)
        #         try:
        #             sg = SendGridAPIClient(sendgrid_api_key)
        #             response = sg.send(message)
        #             print(f"[Inference] Email sent successfully. Status code: {response.status_code}")
        #         except Exception as e:
        #             print(f"[Inference] Failed to send email: {str(e)}")
        # else:
        #     print("[Inference] No email sent (machine condition normal).")
        
        if overall_at_risk:
            # Define workspace_id (hardcoded for now; can be from config or DB)
            workspace_id = os.getenv("WORKSPACE_ID", "default_workspace_id")  # Replace with actual logic to get workspace_id
            
            # Get recipient emails from workspace members
            recipient_emails = self.get_emails_for_workspace(workspace_id)
            if not recipient_emails:
                print(f"[Inference] No emails found for workspace {workspace_id}. Skipping email.")
            else:
                from_email = "thisupun3@gmail.com"
                subject = "Machine Condition Alert"
                body = alert_message
                
                sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
                if not sendgrid_api_key:
                    print("[Inference] SendGrid API Key not found.")
                else:
                    for to_email in recipient_emails:
                        message = Mail(from_email=from_email, to_emails=to_email, subject=subject, plain_text_content=body)
                        try:
                            sg = SendGridAPIClient(sendgrid_api_key)
                            response = sg.send(message)
                            print(f"[Inference] Email sent to {to_email}. Status code: {response.status_code}")
                        except Exception as e:
                            print(f"[Inference] Failed to send email to {to_email}: {str(e)}")
        else:
            print("[Inference] No email sent (machine condition normal).")
        
        return forecast, {"status": "success", "message": alert_message}

    def get_emails_for_workspace(self, workspace_id):
        """
        Retrieve emails of users associated with the given workspace_id.
        """
        if not self.workspaces_collection or not self.users_collection:
            print("[Inference] Database collections not available.")
            return []
        
        # Query workspaces collection for the workspace_id
        workspace_doc = self.workspaces_collection.find_one({"workspace_id": workspace_id})
        if not workspace_doc or "members" not in workspace_doc:
            print(f"[Inference] Workspace {workspace_id} not found or has no members.")
            return []
        
        emails = []
        for member in workspace_doc["members"]:
            user_id = member.get("user")  # Assuming 'user' field holds user ID
            if user_id:
                # Query users collection for email
                user_doc = self.users_collection.find_one({"_id": user_id})  # Assuming _id is the user ID
                if user_doc and "email" in user_doc:
                    emails.append(user_doc["email"])
        
        return emails