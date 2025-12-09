import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
import mlflow

load_dotenv()


def ensure_tracking_uri():
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI not set in environment (.env)")
    mlflow.set_tracking_uri(uri)
    return uri


def upload_and_register(model_path: Path, model_name: str, artifact_subpath: str = "weights"):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"\nUploading '{model_path.name}' as '{model_name}' to MLflow ({mlflow.get_tracking_uri()})")

    # Start a run to attach artifacts to
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f" - MLflow run id: {run_id}")

        # Log the raw model file as an artifact under artifact_subpath (use 'weights' for YOLOModelManager)
        mlflow.log_artifact(str(model_path), artifact_path=artifact_subpath)
        artifact_path_on_run = f"{artifact_subpath}/{model_path.name}"
        artifact_uri = f"runs:/{run_id}/{artifact_path_on_run}"
        print(f" - Artifact uploaded to: {artifact_uri}")

        # Attempt to register the artifact as a model (model registry must be available)
        try:
            print(f" - Registering model '{model_name}' in Model Registry...")
            mv = mlflow.register_model(artifact_uri, model_name)
            print(f" ✓ Registered model '{model_name}' (name={mv.name}, version={mv.version})")
        except Exception as e:
            print(f" ! Could not register model in registry: {e}")
            print("   The artifact is still available in the run artifacts. Use the run id to reference it.")

        return run_id


def parse_models_arg(models_arg):
    # Accepts strings like: path1:ModelName path2:ModelName2
    models = []
    for item in models_arg:
        if ":" in item:
            path_str, name = item.split(":", 1)
        else:
            path_str, name = item, None
        models.append((Path(path_str.strip()), name.strip() if name else None))
    return models


def main():
    parser = argparse.ArgumentParser(description="Upload and register PyTorch models to MLflow (DagsHub)")
    parser.add_argument("--models", "-m", nargs="+", help="List of model mappings: <path>:<model_name>")
    args = parser.parse_args()

    # Defaults (artifact path used by YOLOModelManager is 'weights')
    default_pairs = [
        (Path("ball-yolov8s.pt"), "yolo-ball-detector"),
        (Path("yolov8s.pt"), "yolo-batsman-detector"),
    ]

    try:
        tracking_uri = ensure_tracking_uri()
        print(f"MLflow tracking URI: {tracking_uri}")
    except Exception as e:
        print(f"Error: {e}")
        return

    if args.models:
        raw = parse_models_arg(args.models)
        # fill missing names with defaults if possible
        models = []
        for i, (p, n) in enumerate(raw):
            if not n and i < len(default_pairs):
                n = default_pairs[i][1]
            models.append((p, n))
    else:
        models = default_pairs

    results = {}
    for path, name in models:
        try:
            run_id = upload_and_register(path, name)
            results[path.name] = {"status": "success", "run_id": run_id, "model_name": name}
        except Exception as e:
            results[path.name] = {"status": "failed", "message": str(e)}

    print("\nSummary:\n")
    for k, v in results.items():
        print(f" - {k}: {v}")

    print("\nNext steps:")
    print(" - If registration succeeded, note the model name and version shown in MLflow UI (DagsHub).")
    print(" - Update runtime config (e.g., weights_config.py or .env MODEL_RUN_ID) to point to the run/model you want to use in inference.")


if __name__ == "__main__":
    main()
