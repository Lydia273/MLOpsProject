from google.cloud import storage
import os


def download_data(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

if __name__ == "__main__":
    # Example usage
    bucket_name = "your-bucket-name"
    source_blob_name = "path/to/your/datafile"
    destination_file_name = "data/raw/datafile"
    download_data(bucket_name, source_blob_name, destination_file_name)