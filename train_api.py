"""Minimal FastAPI app to trigger YOLO training and update runtime weight path.

Endpoints:
 - POST /train/yolo -> start YOLO training (background). Accepts JSON body with keys: data, weights, epochs, name
 - GET /weights -> return current configured weights

Note: This is a simple orchestrator. In production you'd want better process
management, persistence of the weight path, and authentication.
"""
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import subprocess
import os
from pathlib import Path
import time
from weights_config import set_yolo_ball_weights, YOLO_BALL_WEIGHTS

app = FastAPI()

class YoloTrainRequest(BaseModel):
    weights: str = 'yolov8s.pt'
    data: str = 'cricket_ball_data/data.yaml'
    epochs: int = 50
    batch: int = 16
    imgsz: int = 640
    name: str = 'yolov8-transfer'
    device: str = '0'


def _run_training_and_update(req: YoloTrainRequest):
    # Run training as a subprocess. This will block the worker, so we run it in background tasks.
    cmd = [
        'python', 'train_yolo.py',
        '-w', req.weights,
        '-d', req.data,
        '-e', str(req.epochs),
        '-b', str(req.batch),
        '--imgsz', str(req.imgsz),
        '-n', req.name,
        '--device', req.device
    ]
    print('[INFO] Running training:', ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print('[ERROR] Training subprocess failed:', e)
        return

    # After training completes, point runtime config to the best weights produced by Ultralytics
    candidate = Path(f"runs/train/{req.name}/weights/best.pt")
    if candidate.exists():
        set_yolo_ball_weights(str(candidate))
        print(f'[INFO] Updated YOLO ball weights to: {candidate}')
    else:
        print('[WARN] Trained best.pt not found at expected location')


@app.post('/train/yolo')
async def train_yolo(req: YoloTrainRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_training_and_update, req)
    return {"status": "training_started", "run_name": req.name}


@app.get('/weights')
async def get_weights():
    return {"yolo_ball_weights": YOLO_BALL_WEIGHTS}
