import os
import logging
import mlflow
import tensorflow as tf
from typing import Optional

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading and caching of TensorFlow models from MLflow artifacts."""
    
    def __init__(self, tracking_uri: str = None, username: str = None, password: str = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.username = username
        self.password = password
        self._model_cache = {}
        logger.info(f"    🔧 ModelManager initialized with URI: {tracking_uri}")

    def load_model_from_run(self, run_id: str, artifact_path: str = "model"):
        """
        Load TensorFlow model from MLflow run artifacts (DagsHub compatible).
        
        Args:
            run_id: MLflow run ID containing the model
            artifact_path: Path to model artifacts within the run
        
        Returns:
            Loaded TensorFlow model
        """
        key = f"run:{run_id}:{artifact_path}"
        
        # Return cached model if available
        if key in self._model_cache:
            logger.info(f"    💾 Using cached model from run: {run_id}")
            return self._model_cache[key]

        # Load from MLflow run artifacts
        logger.info(f"    📥 Downloading model from MLflow run: {run_id}")
        
        try:
            # Download artifacts
            artifact_uri = f"runs:/{run_id}/{artifact_path}"
            logger.info(f"    🔗 Artifact URI: {artifact_uri}")
            local_path = mlflow.artifacts.download_artifacts(artifact_uri)
            logger.info(f"    📂 Downloaded to: {local_path}")
            
            # Load TensorFlow SavedModel
            logger.info("    🔄 Loading TensorFlow SavedModel...")
            model = tf.saved_model.load(local_path)
            self._model_cache[key] = model
            logger.info(f"    ✓ Model loaded and cached from run {run_id}")
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
