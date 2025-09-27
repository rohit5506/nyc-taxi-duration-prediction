import os
import pickle
import mlflow
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import yaml
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
from functools import wraps

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Model prediction duration')
ACTIVE_PREDICTIONS = Gauge('active_predictions', 'Number of active predictions')
MODEL_PREDICTIONS_TOTAL = Counter('model_predictions_total', 'Total predictions made')
PREDICTION_VALUES = Histogram('prediction_values', 'Distribution of prediction values', buckets=[5, 10, 15, 20, 30, 45, 60, 90])

# Define input schema
class TripInput(BaseModel):
    PULocationID: str
    DOLocationID: str
    trip_distance: float

# Initialize FastAPI app
app = FastAPI(title="NYC Taxi Duration Predictor", version="1.0.0")

# Global variables for model and preprocessor
model = None
dv = None

@app.on_event("startup")
def load_artifacts():
    global model, dv
    
    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    model_uri = f"models:/{params['mlflow']['experiment_name']}/production"
    
    print(f"Loading model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    print("Model loaded successfully.")

    # Load preprocessor
    client = mlflow.tracking.MlflowClient()
    run_id = client.get_latest_versions(params['mlflow']['experiment_name'], stages=["Production"])[0].run_id
    dv_path = client.download_artifacts(run_id=run_id, path="preprocessor/dv.pkl")
    
    with open(dv_path, 'rb') as f_in:
        dv = pickle.load(f_in)
    print("Preprocessor loaded successfully.")

@app.get("/")
def read_root():
    return {"status": "NYC Taxi Prediction API is running", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "preprocessor_loaded": dv is not None
    }

@app.get("/metrics")
def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict_duration(trip: TripInput):
    start_time = time.time()
    ACTIVE_PREDICTIONS.inc()
    
    try:
        # Track prediction time
        pred_start = time.time()
        trip_dict = trip.dict()
        X_trip = dv.transform([trip_dict])
        prediction = model.predict(X_trip)
        predicted_duration = float(prediction[0])
        pred_duration = time.time() - pred_start
        
        # Record metrics
        PREDICTION_DURATION.observe(pred_duration)
        MODEL_PREDICTIONS_TOTAL.inc()
        PREDICTION_VALUES.observe(predicted_duration)
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='success').inc()
        
        return {"predicted_duration_minutes": predicted_duration}
        
    except Exception as e:
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='error').inc()
        raise HTTPException(status_code=400, detail=str(e))
    
    finally:
        total_duration = time.time() - start_time
        REQUEST_DURATION.observe(total_duration)
        ACTIVE_PREDICTIONS.dec()
