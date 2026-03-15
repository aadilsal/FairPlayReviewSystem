## Engines Implementation Details

The `engines` folder contains the core computer vision models and tracking algorithms that drive the ball detection system.

---

### 1. YOLO Ball Detector (`yolo_detect.py`)

This module provides a dual-model wrapper for the YOLOv8 architecture, optimized for hardware acceleration (CUDA) and specialized cricket ball detection.

* **Dual-Model Logic**:
* **Model 1 (Global)**: Used for high-precision global scanning across the frame.
* **Model 2 (Refinement)**: Specifically utilized for high-recall detection within a localized **Region of Interest (ROI)**.


* **`detect_roi`**: A critical method that runs Model 2 on a cropped image and automatically performs **Coordinate Mapping**, adding the crop offsets to return the ball's position in global frame space.
* **Rescue Capability**: The `yolo_detect_ball_lowconf` function allows the pipeline to perform "low-confidence rescues" by overriding standard thresholds to find the ball in difficult conditions (e.g., motion blur).
* **Singleton Pattern**: Ensures that the heavy YOLO models are only loaded into memory once and are shared across all processes via `get_global_yolo_detector()`.

---

### 2. CSRT Tracking (`csrt_tracker.py`)

Implements the **Discriminative Correlation Filter with Channel and Spatial Reliability (CSRT)** to bridge detection gaps.

* **Bidirectional Tracking**:
* **`track_forward`**: Initializes from the last known detection (*pre-anchor*) and tracks into the gap.
* **`track_backward`**: Initializes from the first detection after the gap (*post-anchor*) and tracks in reverse.


* **Agreement & Merging**:
The `agree_and_merge` logic reconciles the two tracking paths:
* **Agreement**: If the two trackers meet with a high **IoU** (Intersection over Union), the results are averaged and tagged as `csrt-agreed`.
* **Edge Suspected**: In occlusion gaps, if the trackers disagree significantly (large Euclidean distance), the frame is flagged for **Kinematic Intersection** in the post-processing stage.


* **Compatibility**: Includes a factory function `_create_csrt_tracker` that handles different OpenCV versions (Standard vs. Legacy namespaces).

---

### Summary of Detection Sources

The engines feed the pipeline with data categorized by the following sources:

| Source | Engine | Logic |
| --- | --- | --- |
| `yolo-anchor` | YOLO Model 1 | Standard high-confidence detection. |
| `yolo-rescue` | YOLO Model 1 | Low-confidence pass within a trajectory corridor. |
| `csrt-agreed` | CSRT | Forward and backward trackers converged on the same point. |
| `edge-suspected` | CSRT | Tracker disagreement during occlusion; requires kinematic resolution. |
