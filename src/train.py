# src/train.py

import mlflow
import pickle
import os
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model(config_path: str):
    """Trains the model and logs everything to MLFlow."""
    
    print("Starting model training...")
    
    # Load project configuration
    with open(config_path) as f:
        params = yaml.safe_load(f)

    # --- MLFlow Setup ---
    mlflow_params = params["mlflow"]
    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_experiment(mlflow_params["experiment_name"])
    
    print(f"Connected to MLFlow tracking URI: {mlflow.get_tracking_uri()}")
    
    # Load processed data
    processed_data_path = params["data"]["processed_path"]
    with open(os.path.join(processed_data_path, "X_train.pkl"), "rb") as f:
        X_train = pickle.load(f)
    with open(os.path.join(processed_data_path, "y_train.pkl"), "rb") as f:
        y_train = pickle.load(f)
    with open(os.path.join(processed_data_path, "X_val.pkl"), "rb") as f:
        X_val = pickle.load(f)
    with open(os.path.join(processed_data_path, "y_val.pkl"), "rb") as f:
        y_val = pickle.load(f)
    
    # Start an MLFlow run
    with mlflow.start_run():
        
        # Log model parameters
        model_config = params["model"]
        mlflow.log_params(model_config["params"])
        print(f"Training model '{model_config['name']}' with params: {model_config['params']}")
        
        # Train the model
        rf = RandomForestRegressor(**model_config["params"])
        rf.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = rf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        print(f"Model evaluation complete. RMSE: {rmse}")

        # Log the model artifact itself
        mlflow.sklearn.log_model(rf, "model")
        
        # Log the DictVectorizer as an artifact so it can be used for inference
        dv_path = os.path.join(processed_data_path, "dv.pkl")
        mlflow.log_artifact(dv_path, artifact_path="preprocessor")

        print("✅ Run finished. Check your MLFlow UI!")

if __name__ == "__main__":
    train_model(config_path="configs/params.yaml")
