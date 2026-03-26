# FairPlayReviewSystem

A comprehensive cricket analysis system for fair play review, featuring advanced ball detection with hybrid tracking, batsman identification and tracking, wicket detection, and pose estimation.

## Features

### API and Match Lifecycle

- FastAPI backend with Supabase integration for auth, matches, reviews, notifications, profile, and detection workflows
- User-scoped match ownership and secure per-user CRUD access
- Match stale-state handling: in-progress matches are auto-completed after 24 hours of inactivity
- Manual completion is still supported using normal match update calls
- Match heartbeat endpoint to keep active matches from being auto-completed while users are actively working
- Automatic notifications when stale matches are auto-completed by the system

### 🏏 Advanced Ball Detection

- **YOLOv8-based Detection**: Custom-trained model for cricket ball detection
- **Hybrid Tracking System**: Combines optical flow, physics-based prediction, and post-processing interpolation
- **Gap Filling**: Handles occlusions, bounces, and bat contact using backwards interpolation
- **Real-time Processing**: Optimized for video analysis with configurable tracking modes

### 🏃‍♂️ Batsman Detection & Tracking

- **Person Detection**: YOLO-based person identification
- **Bat Detection**: Specialized bat recognition
- **Batsman Confirmation**: IoU-based matching of persons and bats
- **Pose Estimation**: Keypoint detection for batsman analysis

### 🎯 Wicket Detection

- **Multi-class Detection**: Stumps, bails, and wicket components
- **High Accuracy**: Fine-tuned for cricket-specific scenarios

### 📊 Pipeline Integration

- **End-to-end Processing**: Frame extraction, detection, tracking, and visualization
- **Metadata Output**: JSON metadata for each frame with detections
- **Video Reconstruction**: Processed frames compiled back to video

## Requirements

- Python 3.9+
- Windows, macOS, or Linux
- GPU recommended (CUDA for YOLO acceleration)
- 8GB+ RAM for video processing

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd FairPlayReviewSystem

# Create virtual environment
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Batsman Tracking (Main Pipeline)

```bash
python main.py --input test_videos/sample.mp4 --fps 30 --person-conf 0.5 --bat-conf 0.1
```

### Backend API

```bash
uvicorn API.main_api:app --reload --host 0.0.0.0 --port 8000
```

Open interactive API docs at:

- http://localhost:8000/docs

### Key API Endpoints (Latest)

- **Analyze video**
  - **POST** `/api/analyze-video?match_id=<id>`
  - **Body**: `multipart/form-data`
    - `video_file`: file (required)
    - `original_decision`: string (required)
  - **Query params (optional)**:
    - `person_conf` (default `0.5`)
    - `bat_conf` (default `0.1`)
    - `pad_conf` (default `0.1`)
    - `iou_thresh` (default `0.05`)
    - `consec_frames` (default `3`)
    - `wicket_conf` (default `0.25`)
    - `preprocess` (default `true`)
    - `fps` (default `30`)
    - `display` (default `true`)

- **Wicket configuration**
  - **GET** `/api/matches/{match_id}/wicket-config`
  - **POST** `/api/matches/{match_id}/wicket-config/auto` (multipart upload; runs in background)
  - **PUT** `/api/matches/{match_id}/wicket-config` (manual override)
  - **Configured semantics (latest)**:
    - `configured = true` **iff** `far_box` is present
    - `near_box` may be missing and the config can still be considered configured

### Match Status Behavior

- Allowed statuses: scheduled, in_progress, completed, cancelled, postponed
- Auto-complete timeout: 24 hours of inactivity while status is in_progress
- Completion metadata columns: completed_by_system, auto_completed_at, completion_reason

### Match Maintenance Endpoints

- POST /api/matches/maintenance/auto-complete?timeout_hours=24 triggers stale in-progress auto-completion for the current user
- POST /api/matches/{match_id}/heartbeat refreshes activity timestamp for an active in-progress match

### Ball Detection Only

```python
from BallDetection.pipeline.ball_detector import detect_ball
import cv2

frame = cv2.imread('frame.jpg')
ball_info = detect_ball(frame=frame, frame_idx=0)
if ball_info:
    print("Ball detected:", ball_info)
```

## Project Structure

```
FairPlayReviewSystem/
├── main.py                          # Main CLI for batsman tracking pipeline
├── detection_pipeline.py            # Integrated detection pipeline
├── detection_pipeline.py            # Ball detection with hybrid tracking
├── BallDetection/
│   ├── ball_detector.py             # Core ball detection with YOLO + hybrid tracking
│   ├── yolo_detect.py               # YOLO wrapper
│   ├── preprocessing.py             # Frame enhancement
│   └── ...
├── BatsmanDetection/
│   ├── person_detector.py           # Person detection
│   ├── bat_detector.py              # Bat detection
│   ├── Batsman_finder.py            # Batsman identification logic
│   ├── Batsman_tracker.py           # KCF-based tracking
│   └── pose_estimator.py            # Pose estimation
├── WicketDetection/
│   └── wicket_detector.py           # Wicket detection
├── utils/
│   ├── frame_extractor.py           # Video to frames
│   ├── video_utils.py               # Frames to video
│   └── visualizer.py                # Result visualization
├── weights/                         # Pre-trained model weights
├── outputs/                         # Processing results
└── requirements.txt                 # Python dependencies
```

## Ball Detection Details

### Detection Methods

1. **YOLOv8 Detection**: Primary detection using custom-trained model
2. **Optical Flow Tracking**: For motion blur scenarios
3. **Physics Prediction**: Projectile motion for occlusions
4. **Post-processing Interpolation**: Backwards gap filling for bounces/deflections

### Configuration

```python
DETECTION_CONFIG = {
    'conf_threshold': 0.2,
    'use_hybrid_tracking': True,
    'optical_flow_quality_threshold': 0.7,
    'physics_prediction_max_frames': 5,
    'gravity_constant': 0.5,
    'use_optical_flow': True,
    # ... more options
}
```

### Output Format

```python
ball_info = {
    "box": [x, y, w, h],
    "conf": float,  # Positive for YOLO, negative for predictions
    "source": "yolo" | "optical_flow" | "physics" | "interpolated_*",
    "velocity": [vx, vy]
}
```

### Post-processing

```python
# Post-processing / trajectory smoothing is implemented in the BallDetection pipeline layer.
# See `BallDetection/pipeline/post_processor.py` for the main post-processing orchestration.
```

## CLI Usage

### Main Pipeline

```bash
python main.py [options]

Options:
  --input, -i          Input video path (required)
  --output, -o         Output directory (default: outputs/frames)
  --fps                Output FPS (default: 30)
  --person-conf        Person detection confidence (default: 0.5)
  --bat-conf           Bat detection confidence (default: 0.1)
  --iou-thresh         IoU threshold for bat-person matching (default: 0.05)
  --consec-frames      Consecutive frames for batsman confirmation (default: 3)
  --wicket-conf        Wicket detection confidence (default: 0.25)
```

### Example

```bash
python main.py -i test_videos/cricket_match.mp4 -o outputs/match1 --fps 25 --person-conf 0.6
```

## Output Structure

```
outputs/
└── frames/
    └── video_name_YYYYMMDD_HHMMSS/
        ├── frame_000001.jpg      # Processed frame with annotations
        ├── frame_000001.json     # Detection metadata
        ├── frame_000002.jpg
        ├── ...
        └── video_name_output.mp4 # Reconstructed video
```

### Metadata Format

```json
{
  "frame_index": 0,
  "tracking_active": true,
  "detections": [
    {
      "label": "Ball",
      "data": {
        "box": [100, 200, 50, 50],
        "conf": 0.85,
        "source": "yolo"
      }
    },
    {
      "label": "Batsman",
      "box": [300, 150, 80, 200],
      "tracked": true
    }
  ]
}
```

## Training Custom Models

### Ball Detection Model

```python
from ultralytics import YOLO

# Train YOLOv8
model = YOLO('yolov8n.pt')
model.train(
    data='cricket_ball_data/data.yaml',
    epochs=50,
    imgsz=640,
    project='weights',
    name='yolov8_cricket_ball'
)
```

### Dataset Structure

```
cricket_ball_data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

## Advanced Usage

### Custom Ball Detection

```python
from BallDetection.pipeline.ball_detector import detect_ball

# Process frames
frame_results = []
for idx, frame in enumerate(video_frames):
    ball_info = detect_ball(frame=frame, frame_idx=idx)
    frame_results.append({
        'frame_idx': idx,
        'position': ball_info['box'][:2] if ball_info else None,
        'conf': ball_info['conf'] if ball_info else 0.0,
        'source': ball_info['source'] if ball_info else 'none'
    })
```

### Visualization

```python
# See `utils/visualizer.py` and `BallDetection/utils/ball_debug_visualizer.py` for
# rendering/debug visualization utilities.
```

## Troubleshooting

### Common Issues

- **CUDA not available**: Install PyTorch with CUDA support
- **Model download fails**: Check internet connection for Ultralytics
- **Memory errors**: Reduce batch size or use CPU mode
- **Tracking fails**: Adjust confidence thresholds

### Performance Tips

- Use GPU for YOLO inference
- Lower FPS for faster processing
- Enable hybrid tracking for better accuracy
- Use post-processing for gap filling

## Dependencies

- OpenCV: Computer vision operations
- Ultralytics YOLOv8: Object detection
- PyTorch: Deep learning framework
- NumPy: Numerical computations
- SciPy: Scientific computing (interpolation)

## License

Academic and demonstration use. Check individual component licenses (Ultralytics, PyTorch, OpenCV).

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## Citation

If used in research, please cite the Ultralytics YOLOv8 paper and relevant computer vision works.
