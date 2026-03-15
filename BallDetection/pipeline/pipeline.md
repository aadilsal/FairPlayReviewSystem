## Pipeline Implementation Details

This section details the primary logic within the `pipeline` directory, covering real-time detection, gap classification, post-processing, and trajectory modeling.

---

### 1. Ball Detector (`ball_detector.py`)

A state-machine controller that manages frame-by-frame processing and coordinate systems.

* **`_init_crop` & `_apply_crop**`: Optimizes performance by slicing frames horizontally (e.g., using `HORIZONTAL_CROP_LEFT_PCT`) if the video is horizontal.
* **`detect`**: The main execution loop. It applies the crop, executes the logic for the current state (**Scanning**, **Validating**, or **Tracking**), and then uses **`remap_to_original`** to shift detection coordinates back to the full frame size.
* **Singleton Pattern**: Provides `get_global_ball_detector()` to ensure a single state-machine instance persists across the application lifecycle.

---

### 2. Gap Classifier (`gap_classifier.py`)

Analyzes sequences of missing detections to determine why the ball was lost.

* **`classify_gaps`**: Scans `ball_infos` for contiguous `None` or "ghost" entries and identifies the nearest valid **Anchors** (pre and post-gap).
* **`_classify_gap_type`**: Applies heuristics to label the gap:
* **Occlusion**: Triggered if the detection confidence dropped significantly (> 0.1) just before the gap.
* **Bounce Adjacent**: Triggered if the vertical velocity ($v_y$) flips sign (positive to negative) between the anchors.
* **Mid-flight**: The default state if no specific heuristic is met.



---

### 3. Post-Processor (`post_processor.py`)

The orchestration engine for the **"Anchor & Rescue"** pipeline, executing a 5-phase recovery sequence.

1. **Trajectory Fitting**: Generates a physical model from high-confidence anchors.
2. **YOLO Rescue**: Attempts to find the ball using ultra-low confidence thresholds within a predicted spatial "corridor."
3. **CSRT Rescue**: Uses bidirectional (forward and backward) **Discriminative Correlation Filter** tracking to bridge gaps.
4. **Kinematic Fallback**: For frames where visual detectors fail, it uses **Pure Projection** or **Edge Resolution** (finding sub-frame intersection points of motion arcs).
5. **Final Smoothing**: Applies a `segment_aware_smooth` that respects the physical discontinuity of a bounce.

---

### 4. Trajectory Modeling (`trajectory.py`)

Handles the mathematical and physical representation of the ball's flight path.

* **SegmentModel**: Represents a single continuous arc using quadratic coefficients for $x$ and $y$:

$$x(t) = at^2 + bt + c$$
$$y(t) = at^2 + bt + c$$

* **Initial Conditions Extraction**: Each segment exposes its initial position, velocity, and acceleration (from the quadratic fit) for use in Kalman filter seeding.

* **Constant-Acceleration Kalman Filter**: The Kalman filter now tracks position, velocity, and acceleration ([x, y, v_x, v_y, a_x, a_y]), matching the quadratic trajectory model. At segment initialization, the filter's acceleration state is seeded from the fitted $a$ coefficients for instant convergence.

* **`find_bounce_frame`**: Detects vertical velocity sign flips ($v_y > 2.0$ to $v_y < -2.0$) to identify where the ball hit the ground.
* **`fit_trajectory`**:
	* If a bounce is detected, it splits the anchors into two groups and fits independent `SegmentModel` arcs.
	* Uses `np.polyfit` for coefficients, defaulting to linear (Degree 1) if fewer than 3 points exist.

* **`predict_position`**: Evaluates the fitted polynomials for a specific `frame_idx` using `np.polyval`.

---
