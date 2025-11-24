import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from mlops.config import load_config
from mlops.inference_yolo import run_inference  # Use YOLO inference
from mlops.utils import allowed_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="MLops Video Inference API")
cfg = load_config()

logger.info("="*60)
logger.info("FastAPI Server Starting")
logger.info(f"Upload directory: {cfg.upload_dir}")
logger.info(f"Results directory: {cfg.results_dir}")
logger.info(f"MLflow URI: {cfg.mlflow_tracking_uri}")
logger.info(f"Model Run ID: {cfg.model_run_id}")
logger.info("="*60)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    logger.info("\n" + "="*60)
    logger.info("🎬 NEW VIDEO UPLOAD REQUEST")
    logger.info("="*60)
    
    filename = file.filename
    logger.info(f"📁 Filename: {filename}")
    logger.info(f"📦 Content type: {file.content_type}")
    
    # Validate file type
    if not allowed_file(filename):
        logger.error(f"❌ Unsupported file type: {filename}")
        raise HTTPException(status_code=400, detail="unsupported_file_type")
    
    logger.info("✓ File type validated")

    # Save uploaded file
    upload_path = cfg.upload_dir / filename
    logger.info(f"💾 Saving to: {upload_path}")
    
    try:
        with open(upload_path, "wb") as f:
            contents = await file.read()
            file_size_mb = len(contents) / (1024 * 1024)
            f.write(contents)
        logger.info(f"✓ File saved ({file_size_mb:.2f} MB)")
    except Exception as e:
        logger.error(f"❌ Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"file_save_failed: {e}")

    # Run inference
    logger.info("🚀 Starting inference pipeline...")
    try:
        result = run_inference(upload_path)
        logger.info("✓ Inference completed")
        logger.info(f"📊 Status: {result.get('status')}")
        
        if result.get('status') == 'success':
            logger.info(f"⏱️  Inference time: {result.get('inference_time')}s")
            logger.info(f"🎯 Frames processed: {result.get('frames_processed')}")
            logger.info(f"🔍 Total detections: {result.get('total_detections')}")
            logger.info(f"📝 MLflow Run ID: {result.get('run_id')}")
        else:
            logger.error(f"❌ Inference failed: {result.get('message')}")
        
        logger.info("="*60 + "\n")
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"❌ Inference pipeline error: {e}", exc_info=True)
        logger.info("="*60 + "\n")
        raise HTTPException(status_code=500, detail=f"inference_error: {e}")


def serve():
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    serve()
