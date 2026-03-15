## Utility Implementation Details

This section outlines the internal logic and configurations for the core utility modules: `ball_detector_helpers.py`, `output.py`, and `config.py`.

---

### 1. Ball Detector Helpers (`ball_detector_helpers.py`)

This module implements the state logic and coordinate mapping for the `BallDetector` class.

#### **State Management Functions**

* **`handle_scanning_state`**: Executes a full-frame YOLO search. If a candidate passes filtering, it resets the Kalman filter, sets the `validation_counter` to 1, and transitions to **STATE_VALIDATING**.
* **`handle_validating_state`**: Performs full-frame detection.
* *Success:* Increments the counter; if it reaches `VALIDATION_FRAMES`, it transitions to **STATE_TRACKING**.
* *Failure:* Resets the detector to **STATE_SCANNING**.


* **`handle_tracking_state`**:
1. **Prediction**: Obtains the next position and velocity from the Kalman filter.
2. **Dynamic ROI**: Calculates a crop area centered on the prediction. Size is determined by:

$$\text{Crop Size} = \text{BASE\_CROP\_SIZE} + (\text{VELOCITY\_FACTOR} \times \text{speed})$$


3. **Detection**: Runs YOLO only within the ROI.
4. **Update**: If missed, it generates a "kalman-ghost" (prediction-based entry). If the `miss_streak` exceeds `MAX_MISS_STREAK`, the detector resets.



#### **Coordinate & Result Processing**

* **`finalize_detection_result`**: Enriches the detection dictionary with the current Kalman state (`interpolated_position`), ROI boxes, frame indices, and miss streaks. It then appends this to the detector's history.
* **`remap_to_original`**: Shifts horizontal coordinates ($x$, $ROI_x$, and $interpolated\_x$) by a given `x_offset` to align detections from a cropped frame back to the original video dimensions.

---

### 2. Output Generation (`output.py`)

Orchestrates the final "Anchor & Rescue" data structure, converting internal metadata into a standardized JSON format.

* **Filtering**: Skips `None` entries or those marked as `ghost: True` to ensure only valid or "rescued" data points remain.
* **Positioning**: Prioritizes the `interpolated_position`. If unavailable, it calculates the center: $(x + w/2, y + h/2)$.
| **Confidence Tiering**:
| Tier | Criteria | Source Examples |
| :--- | :--- | :--- |
| **High** | YOLO anchors $\ge$ threshold | `yolo-anchor` |
| **Med** | Successful rescues, agreed tracking, or parabolic edge intersection | `yolo-rescue`, `csrt`, `edge-suspected` (parabolic) |
| **Low** | Pure physics or uncertain geometric intersections | `kinematic`, `edge-suspected` (uncertain) |
* **Uncertainty**: Only flags **Low** tier detections as `uncertain: True`. Edge-suspected detections solved parabolically are promoted to **Med** and are not flagged as uncertain.

---

### 3. System Configuration (`config.py`)

Centralized parameters governing detection, filtering, and trajectory logic.

#### **Core Configurations**

* **`DETECTION_CONFIG`**: Contains model paths (High Precision vs. High Recall), a `conf_threshold` of **0.2**, and an `iou_threshold` of **0.1**.
* **`FILTERS_CONFIG`**: Manages "gates" for `validator.py`.
* *Area:* Detections must be between **250** and **8000** pixels.
* *Circularity:* Enabled to ensure the object matches a ball's roundness.


* **`ROI & CROP_CONFIG`**: Controls the dynamic search window.
* `BASE_CROP_SIZE`: 200px.
* `VELOCITY_FACTOR`: 2.0 (expands ROI based on speed).
* `MAX_CROP_SIZE`: 800px.


* **`STATE_CONFIG`**:
* `VALIDATION_FRAMES`: 2 consecutive detections to lock tracking.
* `MAX_MISS_STREAK`: 5 frames allowed before resetting to Scanning.


* **`POST_PROCESSOR_CONFIG`**: Defines gap thresholds (Short: 3, Long: 10 frames) and sets the Kalman smoothing window to **5 frames**.
