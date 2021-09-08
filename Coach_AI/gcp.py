import os
import shutil
from google.cloud import storage
from params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION, LOCAL_MODEL_NAME

def upload_model_to_gcp():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{LOCAL_MODEL_NAME}"
    blob = bucket.blob(storage_location)
    blob.upload_from_filename(f"../{LOCAL_MODEL_NAME}")
    
    print("Model uploaded to GCP")

if __name__ == '__main__':

    upload_model_to_gcp()