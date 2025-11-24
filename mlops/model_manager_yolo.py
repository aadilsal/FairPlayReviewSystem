"""Model manager for loading YOLO models from MLflow."""
import os
import logging
import mlflow
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class YOLOModelManager:
    """Manages loading and caching of YOLO models from MLflow artifacts."""
    
    def __init__(self, tracking_uri: str = None, username: str = None, password: str = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.username = username
        self.password = password
        self._model_cache = {}
        logger.info(f"    🔧 YOLOModelManager initialized with URI: {tracking_uri}")

    def load_model_from_run(self, run_id: str, artifact_path: str = "weights"):
        """
        Load YOLO model from MLflow run artifacts.
        
        Args:
            run_id: MLflow run ID containing the model
            artifact_path: Path to model artifacts within the run
        
        Returns:
            Loaded YOLO model
        """
        key = f"run:{run_id}:{artifact_path}"
        
        # Return cached model if available
        if key in self._model_cache:
            logger.info(f"    💾 Using cached model from run: {run_id}")
            return self._model_cache[key]

        # Load from MLflow run artifacts
        logger.info(f"    📥 Downloading YOLO model from MLflow run: {run_id}")
        
        try:
            # Download artifacts
            artifact_uri = f"runs:/{run_id}/{artifact_path}"
            logger.info(f"    🔗 Artifact URI: {artifact_uri}")
            local_path = mlflow.artifacts.download_artifacts(artifact_uri)
            logger.info(f"    📂 Downloaded to: {local_path}")
            
            # Find the .pt file in the downloaded directory
            weights_path = Path(local_path)
            pt_files = list(weights_path.glob("*.pt"))
            
            if not pt_files:
                raise FileNotFoundError(f"No .pt weights file found in {local_path}")
            
            model_file = pt_files[0]
            logger.info(f"    📦 Found weights: {model_file.name}")
            
            # Load YOLO model
            logger.info("    🔄 Loading YOLO model...")
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ImportError("ultralytics package not installed. Run: pip install ultralytics")
            
            model = YOLO(str(model_file))
            self._model_cache[key] = model
            logger.info(f"    ✓ YOLO model loaded and cached from run {run_id}")
            
            # Log model info
            if hasattr(model, 'names'):
                logger.info(f"    📋 Model classes: {model.names}")
            
            return model
        except Exception as e:
            logger.error(f"    ❌ Failed to load model: {e}")
            raise

    def get_model(self, run_id: str = None):
        """
        Get model, loading if not already cached.
        Uses environment variable MODEL_RUN_ID if not provided.
        """
        if run_id is None:
            run_id = os.getenv("MODEL_RUN_ID")
            if not run_id:
                raise ValueError("MODEL_RUN_ID must be set in environment or passed as argument")
        
        return self.load_model_from_run(run_id)
