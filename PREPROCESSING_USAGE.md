# Preprocessing Integration - Usage Guide

## ✅ Integration Complete!

The frame preprocessing module is now fully integrated into the FairPlayReviewSystem pipeline with comprehensive console logging.

## 🎨 What's New

### **Adaptive Frame Preprocessing**
The pipeline now automatically:
- ✅ Assesses frame quality (brightness, blur, contrast)
- ✅ Enhances poor quality frames before detection
- ✅ Adjusts confidence thresholds based on quality
- ✅ Logs detailed processing information
- ✅ Provides statistics summary

## 🚀 Usage

### **Command Line (Recommended)**

#### Basic usage (preprocessing enabled by default):
```bash
python main.py --input test_videos/sample.mp4 --output outputs
```

#### Disable preprocessing:
```bash
python main.py --input test_videos/sample.mp4 --no-preprocessing
```

#### Custom brightness target (for dark videos):
```bash
python main.py --input test_videos/sample.mp4 --target-brightness 0.6
```

#### Full configuration:
```bash
python main.py --input test_videos/sample.mp4 \
    --output outputs \
    --fps 30 \
    --motion-preset balanced \
    --target-brightness 0.5
```

### **Command Line Options**

| Option | Description | Default |
|--------|-------------|---------|
| `--no-preprocessing` | Disable frame preprocessing | Enabled |
| `--target-brightness` | Target brightness (0.3-0.7) | 0.5 |
| `--no-motion-prediction` | Disable motion prediction | Enabled |
| `--motion-preset` | Motion preset (conservative/balanced/aggressive) | balanced |

## 📊 Console Output Explained

### **Pipeline Configuration (Start)**
```
======================================================================
FAIRPLAY REVIEW SYSTEM - PIPELINE CONFIGURATION
======================================================================
Motion Prediction: ✓ Enabled
  └─ Preset: balanced
Frame Preprocessing: ✓ Enabled
  └─ Target Brightness: 0.5
======================================================================
```

### **Per-Frame Processing**
```
[INFO] 🎨 ⚠ frame_000123.jpg: Quality=0.42, Detections=1, Conf=0.23, Time=180.5ms
[INFO]    └─ Enhanced: brightness_normalization, clahe_contrast (25.3ms)
```

**Legend:**
- 🎨 = Frame was preprocessed
- ✓ = Good quality (≥0.6)
- ⚠ = Poor quality (<0.6)
- **Quality** = Overall quality score (0-1)
- **Detections** = Number of balls detected
- **Conf** = Confidence threshold used
- **Time** = Total processing time

**Enhancements:**
- `brightness_normalization` = Gamma correction applied
- `clahe_contrast` = Contrast enhancement applied
- `sharpening` = Unsharp masking applied

### **Statistics Summary (End)**
```
======================================================================
📊 PREPROCESSING STATISTICS SUMMARY
======================================================================
Total Frames Processed: 150
Frames Preprocessed: 67 (44.7%)
Average Quality Score: 0.58
Average Preprocessing Time: 22.3ms
Total Preprocessing Time: 1494ms

Quality Distribution:
  Poor (<0.4):       23 ( 15.3%)
  Fair (0.4-0.6):    44 ( 29.3%)
  Good (0.6-0.8):    65 ( 43.3%)
  Excellent (>0.8):  18 ( 12.0%)
======================================================================
```

## 🔧 Configuration in Code

### **Pipeline Configuration**

Edit `pipeline/main_pipeline.py`:

```python
# Frame preprocessing configuration
ENABLE_PREPROCESSING = True   # Toggle on/off
TARGET_BRIGHTNESS = 0.5       # Target brightness (0.3-0.7)
```

### **Programmatic Usage**

```python
from pipeline.main_pipeline import process_frames_pipeline

# Process frames with preprocessing
process_frames_pipeline(
    frame_paths=['path/to/frame1.jpg', 'path/to/frame2.jpg'],
    enable_motion_prediction=True,
    motion_preset='balanced',
    enable_preprocessing=True,      # Enable preprocessing
    target_brightness=0.5           # Target brightness
)
```

## 🎯 Presets for Different Conditions

### **Indoor/Low Light (Dark Videos)**
```bash
python main.py --input video.mp4 --target-brightness 0.6
```
Or in code:
```python
process_frames_pipeline(frame_paths, enable_preprocessing=True, target_brightness=0.6)
```

### **Outdoor/Bright Conditions**
```bash
python main.py --input video.mp4 --target-brightness 0.45
```

### **High Quality Videos (Disable Preprocessing)**
```bash
python main.py --input video.mp4 --no-preprocessing
```

## 📈 Performance Impact

### **Typical Overhead (per frame)**
- Quality assessment: ~2-5ms
- Preprocessing (when needed): ~20-50ms
- Total detection time: ~150-250ms

### **Adaptive Behavior**
- High quality frames (≥0.6): No preprocessing, ~0-5ms overhead
- Medium quality (0.4-0.6): Selective enhancement, ~15-30ms
- Poor quality (<0.4): Full enhancement, ~30-50ms

### **Expected Preprocessing Rates**
- Good lighting conditions: 10-30% of frames
- Mixed conditions: 30-50% of frames
- Poor lighting: 60-80% of frames

## 🧪 Testing

### **Unit Tests**
```bash
python test_preprocessing.py
```

### **Pipeline Integration Test**
```bash
python test_pipeline_preprocessing.py
```

### **Full Examples**
```bash
python example_adaptive_detection.py
python integration_example.py
```

## 📝 Understanding the Logs

### **Example Session**

```
[INFO] 🎨 Frame Preprocessing: ENABLED
[INFO]    Target Brightness: 0.5
[INFO]    Quality Threshold: 0.6

# Frame 1: Good quality, no preprocessing
[INFO]    ✓ frame_001.jpg: Quality=0.72, Detections=1, Conf=0.25, Time=120.5ms

# Frame 2: Poor quality, preprocessed
[INFO] 🎨 ⚠ frame_002.jpg: Quality=0.38, Detections=1, Conf=0.18, Time=185.3ms
[INFO]    └─ Enhanced: brightness_normalization, clahe_contrast (45.2ms)

# Frame 3: Medium quality, selective enhancement
[INFO] 🎨 ⚠ frame_003.jpg: Quality=0.55, Detections=2, Conf=0.23, Time=165.8ms
[INFO]    └─ Enhanced: brightness_normalization (25.1ms)
```

**Interpretation:**
1. Frame 1: High quality (0.72), no enhancement needed, fast processing
2. Frame 2: Poor quality (0.38), full enhancement applied, lower confidence threshold
3. Frame 3: Medium quality (0.55), brightness only corrected

## 🐛 Troubleshooting

### **Issue: Too many frames being preprocessed**

**Solution:** Increase quality threshold in `pipeline/main_pipeline.py`:
```python
# In detect_ball_on_image_adaptive call
quality_threshold=0.7  # Default is 0.6
```

### **Issue: Preprocessing too aggressive**

**Solution:** Use lower target brightness:
```bash
python main.py --input video.mp4 --target-brightness 0.45
```

### **Issue: Processing too slow**

**Solution:** Disable preprocessing for high-quality videos:
```bash
python main.py --input video.mp4 --no-preprocessing
```

### **Issue: Still missing detections**

**Solution:** Use higher target brightness:
```bash
python main.py --input video.mp4 --target-brightness 0.6
```

## 📚 Related Documentation

- **Full API Reference**: `PREPROCESSING_README.md`
- **Implementation Details**: `PREPROCESSING_IMPLEMENTATION.md`
- **Quick Reference**: `PREPROCESSING_QUICKREF.md`
- **Examples**: `example_adaptive_detection.py`, `integration_example.py`

## ✨ Key Benefits

1. **🎯 Improved Detection**: Better ball detection in poor lighting
2. **📊 Transparent**: Detailed logging shows exactly what's happening
3. **⚡ Efficient**: Only preprocesses frames that need it (adaptive mode)
4. **🔧 Configurable**: Easy to tune for different conditions
5. **📈 Monitored**: Statistics show preprocessing effectiveness

## 🎓 Best Practices

1. **Start with defaults**: Try the default configuration first
2. **Check statistics**: Review the summary to see preprocessing rate
3. **Tune if needed**: Adjust target-brightness based on video conditions
4. **Disable for high quality**: Skip preprocessing for well-lit, sharp videos
5. **Monitor logs**: Watch for 🎨 emoji to see which frames are enhanced

---

**Status:** ✅ Fully integrated and tested
**Created:** November 22, 2025
**Module:** `detection/frame_preprocessing.py`
**Pipeline:** `pipeline/main_pipeline.py`
