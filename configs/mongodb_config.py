# For Python 3.12+

# python configs/mongodb_config.py

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

load_dotenv()

mongo_uri = os.getenv("MONGO_URI")
influx_url = os.getenv("INFLUX_URL")
influx_org = os.getenv("INFLUX_ORG")
influx_token = os.getenv("INFLUX_TOKEN")
influx_bucket = os.getenv("INFLUX_BUCKET")
workspace_id = os.getenv("WORKSPACE_ID", "default_workspace")

# Create a new client and connect to the server
client = MongoClient(mongo_uri, server_api=ServerApi('1'))


def get_influx_client():
    from influxdb_client import InfluxDBClient
    return InfluxDBClient(url=influx_url, token=influx_token, org=influx_org, bucket=influx_bucket)

# Function to get the database and collection
def get_database():
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        # db = client["fyp_hourly_1"]
        db = client["maintenancescheduler_db"]
        return db
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return None

# Export the client for direct use if needed
def get_client():
    return client

if __name__ == "__main__":
    db = get_database()
    if db:
        print("Database connection successful.")
    else:
        print("Database connection failed.")
    client.close()

