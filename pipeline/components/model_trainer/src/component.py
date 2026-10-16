import argparse
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from google.cloud import storage

def train_and_upload(bucket_name, destination_blob_name):
    print("Downloading California Housing Data...")
    data = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    
    print("Saving model locally...")
    joblib.dump(model, "model.pkl")
    
    print(f"Uploading model to gs://{bucket_name}/{destination_blob_name}...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename("model.pkl")
    print("Upload complete.")

if __name__ == '__main__':
    # Parse command-line arguments passed by Vertex AI
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()
    
    train_and_upload(args.bucket_name, args.model_path)
