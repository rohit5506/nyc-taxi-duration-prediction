# src/predict.py

import os
import pickle
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
import yaml

# --- 1. Define input schema ---
class TripInput(BaseModel):
    PULocationID: str
    DOLocationID: str
    trip_distance: float

# --- 2. Initialize FastAPI app ---
app = FastAPI()

# --- 3. Load artifacts at startup ---
# These global variables will be populated when the app starts
model = None
dv = None

@app.on_event("startup")
def load_artifacts():
    global model, dv

    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)

    # Set MLFlow tracking URI
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])

    # Construct model URI for the production model
    model_uri = f"models:/{params['mlflow']['experiment_name']}/production"
    print(f"Loading model from: {model_uri}")

    # Download and load the model
    model = mlflow.sklearn.load_model(model_uri)
    print("Model loaded successfully.")

    # Download and load the preprocessor (DictVectorizer)
    # The client knows how to find the artifact path relative to the model
    client = mlflow.tracking.MlflowClient()
    run_id = client.get_latest_versions(params['mlflow']['experiment_name'], stages=["Production"])[0].run_id
    dv_path = client.download_artifacts(run_id=run_id, path="preprocessor/dv.pkl")
    print(f"Preprocessor downloaded to: {dv_path}")

    with open(dv_path, 'rb') as f_in:
        dv = pickle.load(f_in)
    print("Preprocessor loaded successfully.")

# --- 4. Define prediction endpoint ---
@app.post("/predict")
def predict_duration(trip: TripInput):
    trip_dict = trip.dict()
    X_trip = dv.transform([trip_dict])
    prediction = model.predict(X_trip)
    return {"predicted_duration_minutes": float(prediction[0])}

@app.get("/")
def read_root():
    return {"status": "NYC Taxi Prediction API is running."}
