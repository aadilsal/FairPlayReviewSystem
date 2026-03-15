## Core Implementation Details

The `core` folder contains the mathematical and logical foundations of the system, handling state estimation, physical modeling, and detection validation.

---

### 1. Interpolation & State Estimation (`interpolation.py`)

This module uses a **2D Kalman Filter** with a constant-acceleration model to estimate the ball's state and maintain a smooth trajectory even when visual data is noisy or missing.

* **State Vector**: Tracks a 6D state $(x, y, v_x, v_y, a_x, a_y)$ to model constant acceleration motion (gravity-aware).
* **Segment-Based Seeding**: At segment initialization, the filter's acceleration state is seeded from the quadratic trajectory fit for instant convergence.
* **`segment_aware_smooth`**: This is a critical function for cricket physics. It smooths the trajectory but monitors for a `bounce_frame`.
* When a bounce occurs, the filter performs a **warm reset**: it injects the post-bounce position, velocity, and acceleration derived from the segment's quadratic fit, rather than wiping state. This ensures instant convergence and preserves the sharp, physically accurate **"V" shape** at the bounce.

* **`interpolate_trajectory`**: Provides frame-by-frame position estimates by updating the filter with valid detections and recording the predicted state for frames where the ball is occluded.

---

### 2. Kinematics & Physics Fallbacks (`kinematics.py`)

Provides geometric and parabolic modeling to "rescue" the ball's position when all computer vision methods (YOLO and CSRT) fail.

* **`project_position`**: A physics-only fallback that evaluates the fitted quadratic segments from the `TrajectoryModel` to predict $(x, y)$ coordinates for any given frame index.

* **`find_edge_intersection`**: Resolves "edge-suspected" frames using parabolic intersection.
* When trackers from both ends of a gap disagree, this function now treats the incoming and outgoing paths as quadratic arcs (parabolas) fitted from trajectory segments.
* It solves the intersection by equating the two quadratic models ($a_1 t^2 + b_1 t + c_1 = a_2 t^2 + b_2 t + c_2$), rearranging to $(a_1-a_2)t^2 + (b_1-b_2)t + (c_1-c_2)=0$, and finding real roots with $\text{np.roots}$.
* The intersection frame $t^*$ is selected from valid roots within the gap range, and both models are evaluated at $t^*$, averaging $(x, y)$ for the intersection point.
* If no real intersection exists, it falls back to the midpoint average and flags as uncertain.



---

### 3. Validation & Filtering (`validator.py`)

Acts as the "gatekeeper" to prune false positives (e.g., players' shoes, white stadium seats, or birds) from raw YOLO detections.

#### **Heuristic Filters**

* **`is_ball_circular`**: Calculates the circularity of a detection:

$$C = 4\pi \times \frac{\text{Area}}{\text{Perimeter}^2}$$



A detection is rejected if the circularity is below **0.4**.
* **`is_shoe_like`**: Specifically targets common cricket false positives by checking for high aspect ratios ($w/h > 3.0$) and proximity to the bottom of the frame (where feet are typically located).
* **`is_ball_colored`**: Performs an **HSV mask** check to ensure the detection matches the expected hue (White or Red) of a cricket ball.

#### **Trajectory Validation**

* **`corridor_check`**: Validates low-confidence YOLO detections by ensuring they fall within a specific Euclidean distance (the "spatial corridor") of the predicted physical trajectory.
* **`filter_and_select_ball_detection`**: The master orchestration function. It runs all raw detections through the filter chain and returns only the **single highest-confidence** valid ball candidate.

---

### Summary of State Estimation

| Feature | Method | Purpose |
| --- | --- | --- |
| **Smoothing** | Kalman Filter | Reduces jitter in high-confidence detections. |
| **Occlusion** | Linear Intersection | Resolves paths when the ball is hidden. |
| **Bounce** | Segment Reset | Maintains the sharp "V" change in velocity. |
| **Validation** | Circularity/Color | Prunes false positives like shoes or debris. |

---
