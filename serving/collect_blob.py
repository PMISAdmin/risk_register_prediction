from azure.storage.blob import BlobServiceClient
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define log file path and ensure the directory exists
log_file_path = 'app/serving/risk_register_predict'
log_dir = os.path.dirname(log_file_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')

# Retrieve environment variables
CONNECTION_STRING = os.getenv('AZURE_CONNECTION_STRING')
if not CONNECTION_STRING:
    logging.error("CONNECTION_STRING environment variable is not set.")
    raise ValueError("CONNECTION_STRING environment variable is not set.")

CONTAINER_NAME = "pms-modeling-pickle"

# Connect to Azure Blob Storage
try:
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
except Exception as e:
    logging.error(f"Error initializing BlobServiceClient: {e}")
    raise

def write_log(message):
    logging.info(message)

def download_blob_to_file(blob_name, file_path):
    try:
        blob_client = container_client.get_blob_client(blob_name)
        with open(file_path, "wb") as file:
            blob_data = blob_client.download_blob()
            file.write(blob_data.readall())
        write_log(f"Successfully downloaded blob: {blob_name} to {file_path}")
    except Exception as e:
        write_log(f"Error downloading blob {blob_name}: {e}")

def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_blobs():
    logging.info("Download blobs function called.")

    blobs = [
        ("Classification/Risk Register/best_model.pkl", "app/serving/model/best_model.pkl"),
        ("Classification/Risk Register/data_preprocess.pkl", "app/serving/model/data_preprocess.pkl"),
        ("Classification/Risk Register/vectorizer.pkl", "app/serving/model/vectorizer.pkl"),
        ("Classification/Risk Register/label_encoder.pkl", "app/serving/model/label_encoder.pkl"),
    ]

    # Ensure directories exist for each file
    for _, file_path in blobs:
        ensure_directory_exists(file_path)

    # Download each blob
    for blob_name, file_path in blobs:
        download_blob_to_file(blob_name, file_path)

    logging.info("All blobs have been downloaded.")

if __name__ == "__main__":
    download_blobs()
