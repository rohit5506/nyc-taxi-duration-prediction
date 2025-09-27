# src/predict.py

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pickle
import os
import yaml

# --- 1. Define the input data schema ---
class TripInput(BaseModel):
    PULocationID: str
    DOLocationID: str
    trip_distance: float

# --- 2. Initialize the FastAPI app ---
app = FastAPI()

# --- 3. Load Model and Preprocessor ---
def load_artifacts(config_path="configs/params.yaml"):
    """Loads the MLFlow model and the data preprocessor."""
    with open(config_path) as f:
        params = yaml.safe_load(f)
    
    # Connect to MLFlow and get the latest production model URI
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    
    # This assumes your model is in the 'production' stage in MLFlow
    # Format: "models:/<model_name>/<stage>"
    model_uri = f"models:/{params['mlflow']['experiment_name']}/production"
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except mlflow.exceptions.MlflowException:
        # If 'production' model doesn't exist, fall back to latest version
        model_uri = f"models:/{params['mlflow']['experiment_name']}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        
    # Load the DictVectorizer preprocessor
    processed_path = params["data"]["processed_path"]
    with open(os.path.join(processed_path, "dv.pkl"), "rb") as f_in:
        dv = pickle.load(f_in)
        
    return model, dv

model, dv = load_artifacts()

# --- 4. Define the prediction endpoint ---
@app.post("/predict")
def predict_duration(trip: TripInput):
    """Accepts trip data and returns a duration prediction."""
    
    # Convert input data to a dictionary and then to a feature vector
    trip_dict = trip.dict()
    X_trip = dv.transform([trip_dict])
    
    # Make a prediction
    prediction = model.predict(X_trip)
    
    return {"predicted_duration_minutes": float(prediction[0])}

# A root endpoint for health checks
@app.get("/")
def read_root():
    return {"status": "NYC Taxi Prediction API is running."}

