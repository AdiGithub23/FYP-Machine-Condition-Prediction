# For Python 3.12+

# python configs/mongodb_config.py

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

load_dotenv()

mongo_uri = os.getenv("MONGO_URI")

# Create a new client and connect to the server
client = MongoClient(mongo_uri, server_api=ServerApi('1'))

# Function to get the database and collection
def get_database():
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        db = client["fyp_hourly_1"]
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

