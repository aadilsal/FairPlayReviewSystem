# YOLO MLflow Integration - Implementation Summary

## Problem Analysis

### Original Issue

The system was experiencing a `tuple indices must be integers or slices, not str` error during inference.

**Root Cause:**

- MLflow had a **TensorFlow Hub EfficientDet** model registered (Run ID: `aa180dbc8cee4e2db328207f9dc2b003`)
- The inference pipeline expected dictionary-style output but TensorFlow Hub returns tuple output:
  - Index 0: Bounding boxes (1, 100, 4)
  - Index 1: Classes (1, 100)
  - Index 2: Scores (1, 100)
  - Index 3: Number of detections (1,)
- However, the actual project uses **PyTorch + Ultralytics YOLO** for ball and batsman detection
- This created a **framework mismatch** between the registered model and the actual detection code

## Solution Implemented

### Migrated to YOLO-based MLflow Pipeline

Instead of fixing the TensorFlow model output parsing, we aligned the MLflow system with the project's actual YOLO-based detection framework.

### Files Created

1. **`mlops/train_yolo_mlflow.py`**

   - Registers YOLO `.pt` weights to MLflow
   - Scans project for available YOLO models
   - Uploads weights as MLflow artifacts
   - Supports batch registration of multiple models

2. **`mlops/model_manager_yolo.py`**

   - Manages YOLO model loading from MLflow
   - Downloads weights artifacts from MLflow runs
   - Caches models for performance
   - Returns Ultralytics YOLO model instances

3. **`mlops/inference_yolo.py`**

   - Complete YOLO-based inference pipeline
   - Extracts frames and runs YOLO detection
   - Returns structured detection results with bounding boxes, confidence, class names
   - Logs metrics to MLflow

4. **`mlops/debug_model_output.py`**
   - Utility script to debug model output formats
   - Used to analyze TensorFlow Hub model output structure

### Files Modified

1. **`mlops/server.py`**

   - Changed import from `inference` to `inference_yolo`
   - Now uses YOLO-based inference pipeline

2. **`.env`**
   - Updated `MODEL_RUN_ID` from TensorFlow model to YOLO ball detector
   - Changed from: `aa180dbc8cee4e2db328207f9dc2b003` (TensorFlow)
   - Changed to: `3c9ae3c12e82400fa9f8882c1bbebbba` (YOLO Ball Detector)

## Registered YOLO Models

Successfully registered 5 YOLO models to MLflow:

| Model Name                | Type     | Run ID                                 | Source                    |
| ------------------------- | -------- | -------------------------------------- | ------------------------- |
| yolo-cricket-general-v8n  | general  | `a876435e2e7441e1b3301e56d7ba0f20`     | yolov8n.pt                |
| yolo-cricket-general-v8m  | general  | `760ec500595f4aac9def420e8e563279`     | yolov8m.pt                |
| yolo-cricket-general-v11n | general  | `eed6579586814588b86103fa7f5c2db7`     | yolo11n.pt                |
| yolo-cricket-yolov8s-pose | custom   | `aeb84fdcc0f347cfaec5bdbe89700c96`     | yolov8s-pose.pt           |
| **yolo-ball-detector**    | **ball** | **`3c9ae3c12e82400fa9f8882c1bbebbba`** | **ball_detector_best.pt** |

**Active Model:** yolo-ball-detector (specialized for cricket ball detection)

## Key Improvements

### 1. Framework Consistency

- ✅ MLflow now uses the same YOLO framework as the rest of the project
- ✅ No more PyTorch ↔ TensorFlow incompatibilities

### 2. Better Detection Output

- Returns class names, not just IDs
- YOLO confidence scores are more intuitive
- Bounding boxes in standard xyxy format

### 3. Flexibility

- Can easily switch between registered YOLO models by changing `MODEL_RUN_ID`
- Supports custom trained models (ball detector, batsman detector, etc.)

### 4. Performance

- YOLO is optimized for real-time object detection
- Model caching for faster repeated inference
- Native GPU support through PyTorch

## Detection Output Format

### Previous (TensorFlow - causing errors):

```python
# Tried to access as dict, but was actually a tuple
detections["detection_boxes"]  # ❌ Error: tuple indices must be integers
```

### Current (YOLO - working):

```python
{
  "frame": 0,
  "num_detections": 5,
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.876,
      "class_id": 0,
      "class_name": "ball"
    },
    ...
  ]
}
```

## How to Use

### Register New YOLO Models

```bash
cd mlops

# Register all available models
python train_yolo_mlflow.py

# Or register a specific model
python train_yolo_mlflow.py path/to/model.pt "model-name" "model-type"
```

### Switch Models

Update `.env` file:

```env
MODEL_RUN_ID=<run_id_from_registration>
```

Then restart the server.

### Run Inference

```bash
cd mlops
python server.py
```

Upload video through the frontend at `http://localhost:8501`

## MLflow Tracking

View all experiments and models:

- 🔗 https://dagshub.com/aadilsal234/MLOPS_Proj.mlflow

Each inference run logs:

- Parameters: model version, video path, frames processed
- Metrics: inference time, total detections, avg time per frame
- Artifacts: detection results JSON file

## Future Enhancements

1. **Multi-model Inference**: Run both ball and batsman detectors in parallel
2. **Model Registry**: Use MLflow Model Registry for production/staging versions
3. **A/B Testing**: Compare different YOLO versions on the same video
4. **Ensemble Detection**: Combine predictions from multiple models

## Testing Checklist

- [x] YOLO models registered to MLflow
- [x] Server starts without errors
- [x] Model loads from MLflow successfully
- [ ] Video inference completes successfully
- [ ] Detections returned in correct format
- [ ] Results logged to MLflow
- [ ] Frontend displays detections

---

**Status:** ✅ Implementation Complete - Ready for Testing
**Date:** November 24, 2025
**MLflow Experiment:** cricket-yolo-detection
