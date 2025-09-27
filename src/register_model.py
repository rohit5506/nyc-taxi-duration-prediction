# src/register_model.py
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "nyc-taxi-trip-duration"

# --- IMPORTANT: Update this with your MLFlow tracking URI ---
tracking_uri = "http://34.170.162.73:5000"
mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient()

def register_latest_model(experiment_name):
    """Finds the latest run in an experiment and promotes its model to Production."""
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise Exception(f"Experiment '{experiment_name}' not found.")

    # Get runs sorted by start time to find the latest one
    runs = client.search_runs(experiment.experiment_id, order_by=["attributes.start_time DESC"])
    if not runs:
        raise Exception(f"No runs found for experiment '{experiment_name}'.")

    latest_run = runs[0]
    run_id = latest_run.info.run_id
    print(f"Found latest run with ID: {run_id}")

    # Register the model from the latest run
    model_source = f"runs:/{run_id}/model"
    try:
        model_version = client.create_model_version(
            name=MODEL_NAME,
            source=model_source,
            run_id=run_id
        )
        print(f"Registered new model version: {model_version.version}")

        # Give MLFlow a moment to process the registration
        import time
        time.sleep(5)

        # Transition the new version to the "Production" stage
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=model_version.version,
            stage="Production"
        )
        print(f"Model version {model_version.version} is now in 'Production'.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    register_latest_model("nyc-taxi-trip-duration")
