## Cricket Ball Detection – YOLOv8 + Classic CV

A simple pipeline to detect and visualize a cricket ball in images or videos using Ultralytics YOLOv8 (fastest) and classic computer vision (sliding window with color optimization).

### Requirements

- Python 3.9+ (recommended)
- Windows, macOS, or Linux
- GPU optional (CUDA recommended for speed)

### Setup

```bash
# Clone
git clone <your-repo-url>
cd "APNA PROJECT"

# Create and activate venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Or (macOS/Linux)
# python3 -m venv .venv
# source .venv/bin/activate

# Install deps
pip install -r requirements.txt
```

### Quick Start

- Fastest test (YOLO direct on a video):

```bash
python main.py -i test_videos/vid1.mp4 -m yolo_direct --fps 2
```

- Single image test:

```bash
python main.py -i outputs/frames/frame_0000.jpg -m yolo_direct
```

- Classic but still quick (color optimized):

```bash
python main.py -i test_videos/vid1.mp4 -m color_optimized --fps 2
```

### CLI Usage

```bash
python main.py --input <path> [--methods <list>] [--fps <int>] [--model <weights>] [--output <dir>] [--report]
```

- **--input/-i**: image or video path
- **--methods/-m**: one or more of `sliding_window`, `color_optimized`, `yolo_direct`
- **--fps**: frame extraction fps for video (lower = faster overall)
- **--model**: YOLO weights path (default `yolov8n.pt` – auto-downloads)
- **--output/-o**: output root directory (default `outputs`)
- **--report**: save a run summary report

Examples:

```bash
# YOLO direct (fastest end-to-end)
python main.py -i test_videos/vid1.mp4 -m yolo_direct --fps 2

# Sliding window (classic CV)
python main.py -i test_videos/vid1.mp4 -m sliding_window --fps 2

# Run multiple methods in one go
python main.py -i test_videos/vid1.mp4 -m yolo_direct color_optimized --fps 2 --report
```

### Output Locations

- Processed frames: `outputs/video/<method>/processed_frame_*.jpg`
- Composited video: `outputs/video/output_<method>.mp4`
- Single image results: `outputs/images/<name>_<method>.jpg`
- Report (if `--report`): `outputs/detection_report.txt`

### Project Structure

- `main.py`: CLI and pipeline orchestration
- `frame_extractor.py`: video → frames
- `ball_detect.py`: detection logic (YOLO direct and sliding window)
- `outputs/`: results (kept via `.gitkeep`, ignored by git otherwise)
- `test_videos/`: sample videos (ignored by git)
- `requirements.txt`: Python dependencies

### Methods

- **yolo_direct**: Runs YOLOv8 on full frame. Fastest and simplest.
- **color_optimized**: Sliding window constrained by color segmentation (faster than full sliding window).
- **sliding_window**: Full classic sliding window + YOLO on windows (slowest, for experimentation).

### Tips & Troubleshooting

- GPU check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

- If Ultralytics downloads weights on first run, allow network access.
- If OpenCV fails to show windows in headless envs, rely on saved images/videos under `outputs/`.
- If performance is slow, lower `--fps` for video or prefer `-m yolo_direct`.

### License

For academic/demo use. Review upstream licenses for Ultralytics, PyTorch, and OpenCV.
