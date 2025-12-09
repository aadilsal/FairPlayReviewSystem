import os
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


@dataclass
class Config:
    dagshub_username: str
    dagshub_pat: str
    mlflow_tracking_uri: str
    mlflow_username: str
    mlflow_password: str
    mlflow_model_name: str
    model_run_id: str
    model_run_id_ball: str
    model_run_id_batsman: str
    model_version: str
    upload_dir: Path
    results_dir: Path
    max_video_size_mb: int
    device: str


def load_config() -> Config:
    base = Path(__file__).resolve().parents[1]
    upload_dir = Path(os.getenv("UPLOAD_DIR", base / "mlops" / "data" / "uploads"))
    results_dir = Path(os.getenv("RESULTS_DIR", base / "mlops" / "data" / "results"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_run_id = os.getenv("MODEL_RUN_ID", "")
    model_run_id_ball = os.getenv("MODEL_RUN_ID_BALL", model_run_id)
    model_run_id_batsman = os.getenv("MODEL_RUN_ID_BATSMAN", "")

    return Config(
        dagshub_username=os.getenv("DAGSHUB_USERNAME", ""),
        dagshub_pat=os.getenv("DAGSHUB_PAT", ""),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", ""),
        mlflow_username=os.getenv("MLFLOW_TRACKING_USERNAME", ""),
        mlflow_password=os.getenv("MLFLOW_TRACKING_PASSWORD", ""),
        mlflow_model_name=os.getenv("MLFLOW_MODEL_NAME", "default-model"),
        model_run_id=model_run_id,
        model_run_id_ball=model_run_id_ball,
        model_run_id_batsman=model_run_id_batsman,
        model_version=os.getenv("MODEL_VERSION", "production"),
        upload_dir=upload_dir,
        results_dir=results_dir,
        max_video_size_mb=int(os.getenv("MAX_VIDEO_SIZE", 500)),
        device=os.getenv("DEVICE", "cpu"),
    )
