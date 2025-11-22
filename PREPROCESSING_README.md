# Frame Preprocessing Module for Robust Ball Detection

Comprehensive preprocessing module to improve ball detection under poor image quality conditions (brightness, blur, contrast issues).

## 📋 Overview

This module provides:
- **Quality Assessment**: Detect brightness, blur, and contrast issues
- **Image Enhancement**: Adaptive brightness normalization, CLAHE, and sharpening
- **Adaptive Detection**: Quality-based preprocessing with confidence adjustment
- **Performance Monitoring**: Track preprocessing time and effectiveness

## 🚀 Quick Start

### Basic Usage

```python
from detection.detect_balls import detect_ball_on_image_adaptive
import cv2

# Load image
frame = cv2.imread("path/to/frame.jpg")

# Detect with adaptive preprocessing
result = detect_ball_on_image_adaptive(
    image=frame,
    model_path="yolov8n.pt",
    conf=0.25,
    enable_preprocessing=True,
    target_class_names=['sports ball']
)

print(f"Detections: {len(result['detections'])}")
print(f"Quality Score: {result['quality_score']:.2f}")
print(f"Was Preprocessed: {result['was_preprocessed']}")
print(f"Enhancements: {result['enhancements_applied']}")
```

## 📊 Quality Assessment Functions

### 1. Detect Brightness Level

```python
from detection.frame_preprocessing import detect_brightness_level

brightness = detect_brightness_level(frame)
# Returns: "too_dark", "too_bright", or "normal"
```

**Parameters:**
- `frame`: Input frame (BGR or grayscale)
- `dark_threshold`: Below this (0-1) is too dark (default: 0.3)
- `bright_threshold`: Above this (0-1) is too bright (default: 0.7)

### 2. Detect Blur Level

```python
from detection.frame_preprocessing import detect_blur_level

blur_score = detect_blur_level(frame)
# Returns: 0-1, where 0=sharp, 1=very blurry
```

Uses Laplacian variance method to detect edges. Lower variance = more blur.

**Interpretation:**
- `< 0.3`: Sharp, good quality
- `0.3-0.5`: Moderate blur, acceptable
- `> 0.5`: High blur, needs enhancement

### 3. Get Overall Quality Score

```python
from detection.frame_preprocessing import get_frame_quality_score

quality_score, metrics = get_frame_quality_score(frame)
# Returns: (score, QualityMetrics object)
```

**QualityMetrics contains:**
- `brightness_level`: "too_dark", "too_bright", or "normal"
- `brightness_value`: 0-1 scale
- `blur_score`: 0-1 (higher = blurrier)
- `contrast_score`: 0-1 (higher = better)
- `overall_quality`: 0-1 (higher = better)
- `needs_enhancement`: Boolean

## 🔧 Image Enhancement Functions

### 1. Normalize Brightness

```python
from detection.frame_preprocessing import normalize_brightness

enhanced = normalize_brightness(frame, target_brightness=0.5)
```

**Parameters:**
- `target_brightness`: Desired brightness 0-1 (default: 0.5)
- `method`: "gamma" or "linear" (default: "gamma")

Gamma correction provides more natural results than linear scaling.

### 2. Enhance Contrast (CLAHE)

```python
from detection.frame_preprocessing import enhance_contrast

enhanced = enhance_contrast(frame, clip_limit=2.0)
```

**Parameters:**
- `clip_limit`: Contrast limiting threshold (default: 2.0)
- `tile_grid_size`: Grid size for histogram equalization (default: (8, 8))

CLAHE (Contrast Limited Adaptive Histogram Equalization) prevents over-amplification of noise.

### 3. Reduce Blur (Sharpening)

```python
from detection.frame_preprocessing import reduce_blur

enhanced = reduce_blur(frame, strength=1.0, method="unsharp")
```

**Parameters:**
- `strength`: 0.5-2.0, where 1.0 is normal (default: 1.0)
- `method`: "unsharp" or "kernel" (default: "unsharp")

Unsharp masking: `original + (original - blurred) × strength`

### 4. Preprocess Frame (Combined)

```python
from detection.frame_preprocessing import preprocess_frame

enhanced, enhancements = preprocess_frame(
    frame,
    target_brightness=0.5,
    apply_clahe=True,
    apply_sharpening=True,
    sharpening_strength=1.0,
    adaptive=True  # Only apply when needed
)
```

**Returns:**
- `enhanced`: Enhanced frame
- `enhancements`: List of applied enhancements (e.g., ["brightness_normalization", "clahe_contrast"])

## 🎯 Adaptive Detection

### Using AdaptiveBallDetector Class

```python
from detection.frame_preprocessing import AdaptiveBallDetector
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Create detector
detector = AdaptiveBallDetector(
    model=model,
    enable_preprocessing=True,
    quality_threshold=0.6,
    base_confidence=0.25,
    min_confidence=0.15,
    target_brightness=0.5,
    log_enhancements=True
)

# Detect on frame
result = detector.detect(
    frame=frame,
    frame_id="frame_001",
    imgsz=640,
    device='cpu'
)

print(f"Boxes: {result['boxes']}")
print(f"Confidences: {result['confidences']}")
print(f"Quality: {result['quality_score']:.2f}")
print(f"Preprocessed: {result['was_preprocessed']}")
```

### How Adaptive Detection Works

1. **Assess Quality**: Calculate brightness, blur, and contrast scores
2. **Decide Preprocessing**: Apply if quality < threshold or specific issues detected
3. **Enhance Frame**: Apply brightness, contrast, and sharpening as needed
4. **Adjust Confidence**: Lower threshold for poor quality frames
5. **Run Detection**: YOLO inference on preprocessed frame
6. **Return Metadata**: Include quality metrics and preprocessing info

### Confidence Threshold Adjustment

The detector automatically adjusts confidence thresholds based on quality:

| Quality Score | Confidence Multiplier | Example (base=0.25) |
|---------------|----------------------|---------------------|
| ≥ 0.7 (High)  | 1.0×                | 0.25                |
| 0.5-0.7 (Med) | 0.9×                | 0.225               |
| < 0.5 (Low)   | 0.7× (min 0.15)     | 0.175               |

This prevents missing detections in challenging conditions.

## 📈 Statistics and Monitoring

```python
# Get preprocessing statistics
stats = detector.get_statistics()

print(f"Total Frames: {stats['total_frames_processed']}")
print(f"Preprocessed: {stats['frames_preprocessed']} ({stats['preprocessing_rate']*100:.1f}%)")
print(f"Avg Quality: {stats['average_quality_score']:.2f}")
print(f"Avg Preprocessing Time: {stats['average_preprocessing_time_ms']:.1f}ms")

# Reset statistics
detector.reset_statistics()
```

## 🎨 Visualization

### Quality Assessment Visualization

```python
from detection.frame_preprocessing import visualize_quality_assessment

_, metrics = get_frame_quality_score(frame)
vis_frame = visualize_quality_assessment(frame, metrics, show_metrics=True)

cv2.imwrite("quality_visualization.jpg", vis_frame)
```

### Before/After Comparison

```python
from detection.frame_preprocessing import compare_preprocessing

results = compare_preprocessing(frame, show_original=True, show_enhanced=True)

cv2.imwrite("original.jpg", results['original'])
cv2.imwrite("enhanced.jpg", results['enhanced'])
cv2.imwrite("comparison.jpg", results['comparison'])  # Side-by-side
```

## ⚙️ Configuration Examples

### Scenario 1: Very Dark Images (Night/Indoor)

```python
enhanced = normalize_brightness(frame, target_brightness=0.6)  # Brighter
enhanced = enhance_contrast(enhanced, clip_limit=3.0)  # More contrast
```

### Scenario 2: Outdoor/Bright Conditions

```python
enhanced = normalize_brightness(frame, target_brightness=0.45)  # Slightly darker
enhanced = enhance_contrast(enhanced, clip_limit=1.5)  # Less aggressive
```

### Scenario 3: High Motion Blur

```python
enhanced = reduce_blur(frame, strength=1.5, method="unsharp")  # Aggressive sharpening
```

### Scenario 4: Low Contrast (Foggy/Hazy)

```python
enhanced = enhance_contrast(frame, clip_limit=3.0, tile_grid_size=(16, 16))
```

### Scenario 5: Disable Preprocessing

```python
result = detect_ball_on_image_adaptive(
    image=frame,
    enable_preprocessing=False  # Skip all preprocessing
)
```

## 📁 File Structure

```
detection/
├── frame_preprocessing.py      # Main preprocessing module
├── detect_balls.py             # Updated with adaptive detection
└── __init__.py

example_adaptive_detection.py  # Comprehensive examples
PREPROCESSING_README.md         # This file
```

## 🧪 Running Examples

```bash
python example_adaptive_detection.py
```

Available examples:
1. **Single Image Detection**: Compare standard vs adaptive detection
2. **Quality Assessment**: Analyze frame quality metrics
3. **Before/After Comparison**: Visualize preprocessing effects
4. **Batch Processing**: Process multiple frames with monitoring
5. **Custom Preprocessing**: Configure for specific conditions
6. **Performance Benchmark**: Measure preprocessing overhead

## 🔬 Performance Considerations

### Preprocessing Time (typical on CPU)

| Operation                  | Time per Frame | Impact      |
|----------------------------|----------------|-------------|
| Quality Assessment         | ~2-5ms         | Negligible  |
| Brightness Normalization   | ~5-10ms        | Low         |
| CLAHE Contrast            | ~10-20ms       | Moderate    |
| Sharpening                | ~5-15ms        | Low         |
| **Total (all combined)**  | **~20-50ms**   | **Moderate**|

### Optimization Tips

1. **Use Adaptive Mode**: Only preprocess when quality < threshold
2. **Batch Processing**: Amortize model loading overhead
3. **GPU Acceleration**: Focus on YOLO inference (bigger impact)
4. **Skip for High Quality**: Check quality first, skip if good
5. **Parallel Processing**: Process frames in parallel (multi-threading)

### Expected Preprocessing Rate

In typical scenarios:
- **Good lighting, sharp images**: 10-20% of frames need preprocessing
- **Mixed conditions**: 30-50% of frames need preprocessing
- **Poor conditions**: 60-80% of frames need preprocessing

## 📊 Quality Thresholds Reference

### Brightness (0-1 scale)
- **Too Dark**: < 0.3
- **Optimal**: 0.4-0.6
- **Too Bright**: > 0.7

### Blur Score (0-1 scale)
- **Sharp**: < 0.3
- **Acceptable**: 0.3-0.5
- **Blurry**: > 0.5

### Contrast Score (0-1 scale)
- **Poor**: < 0.3
- **Acceptable**: 0.3-0.5
- **Good**: > 0.5

### Overall Quality (0-1 scale)
- **Poor**: < 0.4
- **Fair**: 0.4-0.6
- **Good**: 0.6-0.8
- **Excellent**: > 0.8

## 🔧 Integration with Existing Code

### Option 1: Replace Existing Function

```python
# OLD CODE:
from detection.detect_balls import detect_ball_on_image
detections = detect_ball_on_image(frame, model_path="yolov8n.pt")

# NEW CODE (drop-in replacement):
from detection.detect_balls import detect_ball_on_image_adaptive
result = detect_ball_on_image_adaptive(frame, model_path="yolov8n.pt")
detections = result['detections']  # Same format
```

### Option 2: Conditional Preprocessing

```python
from detection.frame_preprocessing import get_frame_quality_score, preprocess_frame
from detection.detect_balls import detect_ball_on_image

# Check quality first
quality_score, _ = get_frame_quality_score(frame)

if quality_score < 0.6:
    # Preprocess poor quality frames
    frame, _ = preprocess_frame(frame)

# Run detection
detections = detect_ball_on_image(frame, model_path="yolov8n.pt")
```

### Option 3: Batch Processing with Monitoring

```python
from detection.frame_preprocessing import AdaptiveBallDetector
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
detector = AdaptiveBallDetector(model, enable_preprocessing=True)

all_detections = []
for frame in frames:
    result = detector.detect(frame)
    all_detections.extend(result['boxes'])

# Print statistics
stats = detector.get_statistics()
print(f"Preprocessed {stats['preprocessing_rate']*100:.1f}% of frames")
```

## 🐛 Troubleshooting

### Issue: Preprocessing too aggressive

**Solution**: Reduce enhancement strength
```python
enhanced, _ = preprocess_frame(
    frame,
    target_brightness=0.5,  # Try 0.45 or 0.55
    sharpening_strength=0.8  # Reduce from 1.0
)
```

### Issue: Still missing detections

**Solution**: Lower confidence threshold more aggressively
```python
detector = AdaptiveBallDetector(
    model=model,
    base_confidence=0.20,  # Lower from 0.25
    min_confidence=0.10    # Lower from 0.15
)
```

### Issue: False positives increased

**Solution**: Use stricter quality threshold
```python
detector = AdaptiveBallDetector(
    model=model,
    quality_threshold=0.7,  # Higher from 0.6
    base_confidence=0.30    # Higher from 0.25
)
```

### Issue: Processing too slow

**Solution**: Disable preprocessing or use selective mode
```python
# Only preprocess very poor frames
detector = AdaptiveBallDetector(
    model=model,
    quality_threshold=0.4,  # Only preprocess worst frames
    enable_preprocessing=True
)
```

## 📚 Dependencies

Required packages (already in `requirements.txt`):
- `opencv-python` (cv2): Image processing
- `numpy`: Numerical operations
- `ultralytics`: YOLO model

## 🎓 Algorithm Details

### Brightness Detection
Uses average pixel intensity in grayscale:
```python
brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) / 255.0
```

### Blur Detection (Laplacian Variance)
Computes variance of Laplacian edge detection:
```python
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
variance = laplacian.var()
blur_score = 1.0 - min(variance / 500.0, 1.0)
```

### Contrast Enhancement (CLAHE)
Applies histogram equalization to L-channel in LAB color space:
```python
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
l_enhanced = clahe.apply(l)
enhanced = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)
```

### Sharpening (Unsharp Masking)
```python
blurred = cv2.GaussianBlur(frame, (0, 0), 2.0)
sharpened = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
```

## 📝 License

Part of FairPlayReviewSystem project.

## 👥 Author

FairPlayReviewSystem Team

## 📅 Last Updated

November 22, 2025
