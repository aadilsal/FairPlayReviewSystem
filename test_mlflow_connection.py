"""Test MLflow connection to DagsHub."""
import os
from dotenv import load_dotenv
import mlflow

load_dotenv()

# Set MLflow tracking with authentication
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Set environment variables for authentication (DagsHub method)
os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
os.environ['MLFLOW_TRACKING_USERNAME'] = username
os.environ['MLFLOW_TRACKING_PASSWORD'] = password

mlflow.set_tracking_uri(tracking_uri)

print("Testing MLflow connection to DagsHub...")
print(f"Tracking URI: {tracking_uri}")
print(f"Username: {username}")

# Try to create or get experiment
try:
    # Try to get or create experiment
    experiment_name = "test-connection"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Creating experiment: {experiment_name}")
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    
    # Test logging
    with mlflow.start_run():
        mlflow.log_param("test", "connection")
        mlflow.log_metric("status", 1)
        run_id = mlflow.active_run().info.run_id
        print(f"\n✓ MLflow connection successful!")
        print(f"Run ID: {run_id}")
        print(f"View at: {tracking_uri}")
except Exception as e:
    print(f"\n✗ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("1. Verify your DagsHub repository exists: https://dagshub.com/aadilsal234/FairPlayReviewSystem")
    print("2. Enable MLflow in DagsHub repo settings if not already enabled")
    print("3. Check that your PAT token has the correct permissions")
    raise
