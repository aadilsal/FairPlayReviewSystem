# Ball Detection with Motion Prediction

This feature adds intelligent motion prediction to handle missed ball detections in video sequences. When YOLO fails to detect the ball in certain frames (due to blur, occlusion, or low confidence), the system can predict ball positions using trajectory analysis.

## Features

### 1. **Automatic Gap Filling**
- Identifies consecutive frames without ball detections
- Fills gaps with predicted ball positions
- Configurable maximum gap size

### 2. **Two Prediction Methods**

#### Linear Interpolation (Fast, Simple)
- Best for gaps of 1-3 frames
- Linearly interpolates position between known detections
- Assumes constant velocity
- Very fast and reliable for short gaps

#### Kalman Filtering (Accurate, Smooth)
- Best for gaps of 4+ frames
- Uses physics-based motion model (constant velocity)
- Provides smoother, more accurate predictions
- Accounts for motion uncertainty

### 3. **Visual Distinction**
- Actual detections: **Green boxes**
- Predicted detections: **Yellow boxes** with "(pred)" label
- Clear confidence scores for both types

### 4. **Enhanced CSV Output**
New fields added:
- `detection_type`: "detected" or "predicted"
- `confidence`: Actual YOLO score or configured prediction confidence (default 0.3)

## Quick Start

### Basic Usage

```python
from detection.detect_balls import detect_balls

# Run detection with motion prediction enabled
records = detect_balls(
    frames_dir='test/images',
    model_path='yolov8n.pt',
    output_csv='runs/detections.csv',
    annotated_dir='runs/annotated',
    target_class_names=['sports ball'],
    enable_motion_prediction=True,  # Enable motion prediction
    max_gap_frames=5,                # Fill gaps up to 5 frames
    prediction_confidence=0.3,       # Confidence for predictions
    use_kalman=False,                # Use linear interpolation
)
```

### Advanced: Using Kalman Filter

```python
# For longer gaps or smoother predictions
records = detect_balls(
    frames_dir='test/images',
    model_path='yolov8n.pt',
    output_csv='runs/detections.csv',
    annotated_dir='runs/annotated',
    target_class_names=['sports ball'],
    enable_motion_prediction=True,
    max_gap_frames=10,               # Fill larger gaps
    prediction_confidence=0.3,
    use_kalman=True,                 # Enable Kalman filtering
)
```

### Post-Processing Existing Detections

```python
from detection.ball_tracker import fill_detection_gaps
import csv

# Load existing detections
detections = []
with open('runs/detections.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        detections.append({
            'frame_index': int(row['frame_index']),
            'x_min': float(row['x_min']),
            'y_min': float(row['y_min']),
            'x_max': float(row['x_max']),
            'y_max': float(row['y_max']),
            'confidence': float(row['confidence']),
            'class_name': row['class_name'],
        })

# Fill gaps
filled = fill_detection_gaps(
    detections,
    max_gap_frames=5,
    prediction_confidence=0.3,
    use_kalman=False
)

print(f"Added {len(filled) - len(detections)} predictions")
```

### Manual Tracking

```python
from detection.ball_tracker import BallTracker

# Create tracker
tracker = BallTracker(
    max_gap_frames=5,
    prediction_confidence=0.3,
    use_kalman=False
)

# Process frames
for frame_idx in range(total_frames):
    detection = detect_ball_in_frame(frame_idx)  # Your detection function
    tracker.update(frame_idx, detection)

# Fill gaps
tracker.fill_gaps()

# Get all detections (actual + predicted)
all_detections = tracker.get_all_detections()
```

## Parameters

### `detect_balls()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_motion_prediction` | bool | `True` | Enable/disable motion prediction |
| `max_gap_frames` | int | `5` | Maximum gap size to fill (frames) |
| `prediction_confidence` | float | `0.3` | Confidence score for predictions (0-1) |
| `use_kalman` | bool | `False` | Use Kalman filter for gaps > 3 frames |

### `BallTracker` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_gap_frames` | int | `5` | Maximum consecutive frames to fill |
| `prediction_confidence` | float | `0.3` | Confidence for predicted detections |
| `use_kalman` | bool | `False` | Enable Kalman filtering |
| `kalman_process_noise` | float | `1.0` | Process noise covariance (motion uncertainty) |
| `kalman_measurement_noise` | float | `10.0` | Measurement noise covariance (detection uncertainty) |

## Examples

Run the example script to see all features in action:

```bash
python example_motion_prediction.py
```

This will demonstrate:
1. ✓ Basic detection without prediction
2. ✓ Detection with linear interpolation
3. ✓ Detection with Kalman filtering
4. ✓ Manual tracker usage
5. ✓ Post-processing existing CSV files

## How It Works

### Linear Interpolation

For gaps of 1-3 frames, the system uses simple linear interpolation:

```
Frame 10: Ball at (100, 200)  ← detected
Frame 11: Ball at (105, 205)  ← predicted (interpolated)
Frame 12: Ball at (110, 210)  ← predicted (interpolated)
Frame 13: Ball at (115, 215)  ← detected
```

The bounding box coordinates are linearly interpolated:
```python
x(t) = x1 + t * (x2 - x1)  where t ∈ [0, 1]
```

### Kalman Filtering

For longer gaps (4+ frames), Kalman filtering provides smoother predictions:

1. **State**: `[x, y, vx, vy]` (position + velocity)
2. **Prediction**: Use constant velocity model
3. **Update**: When detection available, correct prediction
4. **Missing frames**: Continue predicting without update

This accounts for:
- Motion uncertainty (process noise)
- Detection uncertainty (measurement noise)
- Smooth trajectory estimation

### Gap Detection

The system automatically identifies gaps:
```python
Frames with detections: [0, 1, 4, 5, 7, 10, 11]
Gaps found:
  - Frames 2-3 (size: 2)   ← will be filled
  - Frame 6 (size: 1)       ← will be filled
  - Frames 8-9 (size: 2)    ← will be filled
```

## CSV Output Format

Enhanced CSV includes motion prediction information:

```csv
frame_index,frame_id,x_min,y_min,x_max,y_max,confidence,class_id,class_name,detection_type
0,frame_000.jpg,100,200,150,250,0.92,,sports ball,detected
1,frame_001.jpg,105,205,155,255,0.88,,sports ball,detected
2,frame_002.jpg,110,210,160,260,0.30,,sports ball,predicted
3,frame_003.jpg,115,215,165,265,0.85,,sports ball,detected
```

## Visualization

Annotated frames show:
- **Green boxes**: Actual YOLO detections
- **Yellow boxes**: Motion predictions
- **Labels**: Include "(pred)" suffix for predictions

Example:
```
┌─────────────────────────┐
│                         │
│    ┌──────┐            │  ← Yellow box (predicted)
│    │ ball │            │     "sports ball 0.30 (pred)"
│    │ 0.30 │            │
│    └──────┘            │
│                         │
│            ┌──────┐    │  ← Green box (detected)
│            │ ball │    │     "sports ball 0.92"
│            │ 0.92 │    │
│            └──────┘    │
└─────────────────────────┘
```

## Performance Considerations

### Speed
- **Linear interpolation**: Very fast, negligible overhead
- **Kalman filtering**: Slightly slower, still real-time capable

### Accuracy
- **Short gaps (1-3 frames)**: Linear interpolation is sufficient
- **Longer gaps (4-10 frames)**: Kalman filter recommended
- **Very long gaps (>10 frames)**: Consider raising detection confidence threshold instead

### Tuning Tips

1. **`max_gap_frames`**: 
   - Start with 5 for typical video
   - Increase to 10 for high-quality predictions needed
   - Decrease to 2-3 for conservative gap filling

2. **`prediction_confidence`**:
   - 0.3 is a good default (clearly lower than detection threshold)
   - Use 0.1-0.2 for very uncertain predictions
   - Use 0.4-0.5 if predictions are reliable

3. **`use_kalman`**:
   - `False` for fast processing and short gaps
   - `True` for smoother trajectories and longer gaps

## Integration with Existing Code

The motion prediction feature is **backward compatible**. To disable:

```python
records = detect_balls(
    frames_dir='test/images',
    model_path='yolov8n.pt',
    enable_motion_prediction=False,  # Disable
    # ... other parameters
)
```

Old CSV files without `detection_type` column will still work.

## Troubleshooting

### Issue: Too many predictions
**Solution**: Reduce `max_gap_frames` or increase YOLO confidence threshold

### Issue: Predictions are inaccurate
**Solution**: Enable Kalman filtering with `use_kalman=True`

### Issue: Still missing detections
**Solution**: 
1. Check if gaps exceed `max_gap_frames`
2. Lower YOLO detection confidence threshold
3. Improve ball detector model training

### Issue: Predicted boxes in wrong positions
**Solution**: 
1. Verify detection data quality
2. Tune Kalman filter noise parameters
3. Use linear interpolation for erratic motion

## API Reference

### `BallTracker` Class

```python
class BallTracker:
    def __init__(self, max_gap_frames=5, prediction_confidence=0.3, 
                 use_kalman=False, kalman_process_noise=1.0, 
                 kalman_measurement_noise=10.0)
    
    def update(self, frame_index: int, detection: Optional[Dict])
    def fill_gaps(self)
    def get_all_detections(self) -> List[Dict]
    def get_detection(self, frame_index: int) -> Optional[Dict]
    def is_predicted(self, frame_index: int) -> bool
```

### Helper Functions

```python
def fill_detection_gaps(detections, max_gap_frames=5, 
                        prediction_confidence=0.3, use_kalman=False)

def filter_ball_detections(detections, class_name='sports ball')

def group_detections_by_frame(detections)
```

## References

- YOLO: [ultralytics/yolov8](https://github.com/ultralytics/ultralytics)
- Kalman Filter: Constant velocity motion model
- Linear Interpolation: Simple linear extrapolation

## Future Enhancements

Potential improvements:
- [ ] Smooth predictions using forward-backward smoothing
- [ ] Multi-ball tracking with ID assignment
- [ ] Acceleration-aware motion model
- [ ] Deep learning-based trajectory prediction
- [ ] Automatic confidence threshold tuning

## License

Same as parent project.
