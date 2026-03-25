## Cricket Ball Detection & Post-Processing System Architecture

This system implements a multi-stage **"Anchor & Rescue"** pipeline. It combines deep learning, discriminative correlation filters, and kinematic modeling to maintain a continuous ball trajectory through occlusions and bounces.

### 1. Core Detection & State Management

*The entry point manages frame-by-frame processing, coordinate normalization, and state transitions.*

* **`ball_detector.py`**: The primary controller. It handles horizontal center-cropping for performance and remaps coordinates. It utilizes a three-state machine: **Scanning**, **Validating**, and **Tracking**.
* **`ball_detector_helpers.py`**: Implements the logic for each state:
* **Scanning**: Full-frame YOLO search via `yolo_detect.py`.
* **Validating**: Confirms temporal consistency over a fixed frame count.
* **Tracking**: Predicts positions via Kalman Filter and performs localized searches in a **Dynamic ROI** (Region of Interest) scaled by ball velocity.


* **`validator.py`**: Provides geometric and CV "gates" (circularity, color, area, and shoe-like heuristics) to filter false positives before they enter the state machine.

### 2. Trajectory Modeling & Kinematics

*These modules provide the physical "ground truth" used to validate and rescue missing detections.*

* **`trajectory.py`**: Fits degree-2 polynomials to high-confidence detections:

$$y(t) = at^2 + bt + c$$



It detects **bounce frames** by identifying vertical velocity ($v_y$) sign flips, splitting the trajectory into independent segments.
* **`interpolation.py`**: Implements a **2D Kalman Filter**. It features **segment-aware smoothing** that resets at bounce frames to preserve the sharp "V" impact shape.
* **`kinematics.py`**: Provides parabolic projection and a geometric intersection solver to find sub-frame meeting points of motion arcs during occlusions.

### 3. Post-Processing & "Rescue" Pipeline

*This stage fills gaps where the real-time detector failed using a 5-phase sequence.*

1. **Gap Classification (`gap_classifier.py`)**: Labels missing segments as *occlusion*, *bounce_adjacent*, or *mid_flight*.
2. **Low-Conf YOLO Rescue**: Re-runs YOLO at an ultra-low threshold ($conf \approx 0.05$) within a spatial corridor defined by the physics model.
3. **CSRT Tracking (`csrt_tracker.py`)**: Runs bidirectional (forward and backward) tracking from the nearest valid anchors.
4. **Agreement & Merge**: Reconciles trackers; if they disagree during an occlusion, the system flags the frame for kinematic intersection.
5. **Final Smoothing**: Applies the segment-aware Kalman smoother to produce the final continuous path.

### 4. System Support & Output

* **`yolo_detect.py`**: A dual-model wrapper. **Model 1** is optimized for global scanning; **Model 2** is used for high-recall ROI refinement.
* **`config.py`**: Centralized repository for detection thresholds, filter parameters, and physics heuristics.
* **`output.py`**: Formats results into JSON, assigning a **Confidence Tier** (High/Med/Low) based on the data source (e.g., YOLO anchor vs. Kinematic fallback).

---

### Data Flow Summary

1. **Real-Time**: `ball_detector` $\rightarrow$ `yolo_detect` $\rightarrow$ `validator` $\rightarrow$ `ball_infos`.
2. **Modeling**: `ball_infos` $\rightarrow$ `trajectory` (detect bounces/fit arcs).
3. **Rescue**: `gap_classifier` $\rightarrow$ `csrt_tracker` / `kinematics` $\rightarrow$ `ball_infos` (updated).
4. **Finalize**: `interpolation` (smoothing) $\rightarrow$ `output`.

---

### Documentation Links

* [Core Details](core/core.md)
* [Engines Details](engines/engines.md)
* [Pipeline Details](pipeline/pipeline.md)
* [Utils Details](utils/utils.md)

