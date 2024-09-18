from azure.storage.blob import BlobServiceClient
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Azure Blob Storage information
storage_account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
connection_string = os.getenv("AZURE_CONNECTION_STRING")

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = "pms-modeling-pickle"
folder_path = "Classification/Risk Register/"

# File paths
files_to_upload = [
    '/app/data/X_train.pkl',
    '/app/data/y_train.pkl',
    '/app/data/y_test.pkl',
    '/app/model/data_preprocessing/data_preprocess.pkl',
    '/app/model/data_preprocessing/vectorizer.pkl',
    '/app/model/label_encoder/label_encoder.pkl',
    '/app/model/models/best_model.pkl'
]

def upload_files_to_blob(file_paths, connection_string, container_name, folder_path):

    # Create a BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    # Create a container client
    container_client = blob_service_client.get_container_client(container_name)
    
    # Create the container if it does not exist
    try:
        container_client.create_container()
    except Exception as e:
        # If the container already exists, this will catch the exception
        print(f"Container already exists or could not be created: {e}")

    # Upload each file in the list of file paths
    for file_path in file_paths:

        if os.path.isfile(file_path):
            
            filename = os.path.basename(file_path)
            
            # Include the folder path in the blob name
            blob_path = f"{folder_path}{filename}"
            blob_client = container_client.get_blob_client(blob_path)
            
            # Upload the file
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
                print(f"Uploaded {filename} to blob storage at {blob_path}.")
        else:
            print(f"File {file_path} does not exist.")

if __name__ == "__main__":
    upload_files_to_blob(files_to_upload, connection_string, container_name, folder_path)

