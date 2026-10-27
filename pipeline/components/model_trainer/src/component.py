import argparse
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from google.cloud import storage

# Minimum quality a new model must reach before it is uploaded (and then deployed)
MIN_R2 = 0.7

def train_and_upload(bucket_name, destination_blob_name):
    print("Downloading California Housing Data...")
    data = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the 20% test set that was held back
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions) * 100000  # target is in $100,000s
    print(f"Test R2: {r2:.3f}")
    print(f"Test MAE: ${mae:,.0f}")

    # Model validation: only a good enough model is uploaded. Otherwise this step
    # fails, nothing is uploaded, and the model that is currently deployed stays live.
    if r2 < MIN_R2:
        raise SystemExit(f"Model rejected: R2 {r2:.3f} is below the minimum of {MIN_R2}")

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