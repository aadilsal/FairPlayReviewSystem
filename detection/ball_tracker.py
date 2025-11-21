"""Ball tracking with motion prediction to handle missed detections.

This module provides trajectory tracking and motion prediction for ball detection,
filling gaps in detection data using linear interpolation and Kalman filtering.

Key Components:
    - BallTracker: Maintains ball trajectory and predicts positions for missing frames
    - Linear interpolation for short gaps (1-3 frames)
    - Kalman filtering for longer gaps (optional, provides smoother predictions)
    
Usage Example:
    ```python
    from detection.ball_tracker import BallTracker, fill_detection_gaps
    
    # Option 1: Process detection records
    records = [...]  # List of detection dicts with frame_index and bbox
    filled_records = fill_detection_gaps(
        records, 
        max_gap_frames=3,
        prediction_confidence=0.3,
        use_kalman=False
    )
    
    # Option 2: Use tracker directly
    tracker = BallTracker(max_gap_frames=3, use_kalman=False)
    for frame_idx in range(total_frames):
        detection = get_detection_for_frame(frame_idx)  # or None
        tracker.update(frame_idx, detection)
    
    all_detections = tracker.get_all_detections()
    ```
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class KalmanFilter:
    """Simple 2D Kalman filter for ball position and velocity tracking.
    
    State vector: [x, y, vx, vy] (position and velocity)
    Measurement: [x, y] (observed center position)
    
    This filter predicts smooth trajectories even with missing observations,
    making it ideal for handling detection gaps in video sequences.
    """
    
    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 10.0):
        """Initialize Kalman filter.
        
        Args:
            process_noise: Process noise covariance (motion model uncertainty)
            measurement_noise: Measurement noise covariance (detection uncertainty)
        """
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        self.initialized = False
        
        # State transition matrix (constant velocity model)
        # x_new = x + vx, y_new = y + vy
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=float)
        
        # Measurement matrix (we observe x, y only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        
        # Covariance matrix
        self.P = np.eye(4) * 1000  # High initial uncertainty
        
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
    
    def predict(self) -> np.ndarray:
        """Predict next state.
        
        Returns:
            Predicted state [x, y, vx, vy]
        """
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state.copy()
    
    def update(self, measurement: np.ndarray):
        """Update filter with new measurement.
        
        Args:
            measurement: Observed [x, y] position
        """
        if not self.initialized:
            # Initialize state with first measurement
            self.state[0] = measurement[0]
            self.state[1] = measurement[1]
            self.state[2] = 0  # Initial velocity = 0
            self.state[3] = 0
            self.initialized = True
            return
        
        # Innovation (measurement residual)
        y = measurement - (self.H @ self.state)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position estimate.
        
        Returns:
            (x, y) position tuple
        """
        return float(self.state[0]), float(self.state[1])


class BallTracker:
    """Tracks ball trajectory and predicts positions for missing detections.
    
    This tracker maintains a history of ball detections and fills gaps using
    motion prediction. Short gaps (1-3 frames) use linear interpolation for
    speed and simplicity. Longer gaps can optionally use Kalman filtering
    for smoother, physics-aware predictions.
    
    Attributes:
        max_gap_frames: Maximum gap size to fill (frames beyond this are left empty)
        prediction_confidence: Confidence score assigned to predicted detections
        use_kalman: Whether to use Kalman filtering for longer gaps
    """
    
    def __init__(
        self,
        max_gap_frames: int = 5,
        prediction_confidence: float = 0.3,
        use_kalman: bool = False,
        kalman_process_noise: float = 1.0,
        kalman_measurement_noise: float = 10.0
    ):
        """Initialize ball tracker.
        
        Args:
            max_gap_frames: Maximum consecutive frames to fill with predictions
            prediction_confidence: Confidence score for predicted detections (0-1)
            use_kalman: Use Kalman filter for gaps > 3 frames
            kalman_process_noise: Process noise for Kalman filter
            kalman_measurement_noise: Measurement noise for Kalman filter
        """
        self.max_gap_frames = max_gap_frames
        self.prediction_confidence = prediction_confidence
        self.use_kalman = use_kalman
        
        # Detection history: frame_index -> detection dict
        self.detections: Dict[int, Dict[str, Any]] = {}
        
        # Track which detections are predicted
        self.predicted_frames: set = set()
        
        # Kalman filter (initialized on first detection)
        self.kalman: Optional[KalmanFilter] = None
        if use_kalman:
            self.kalman = KalmanFilter(kalman_process_noise, kalman_measurement_noise)
    
    def update(self, frame_index: int, detection: Optional[Dict[str, Any]]):
        """Update tracker with detection for a frame.
        
        Args:
            frame_index: Frame number
            detection: Detection dict with keys: x_min, y_min, x_max, y_max, confidence
                       Or None if no detection in this frame
        """
        if detection is not None:
            # Store actual detection
            self.detections[frame_index] = detection.copy()
            self.detections[frame_index]['detection_type'] = 'detected'
            
            # Update Kalman filter if using it
            if self.kalman is not None:
                center = self._get_bbox_center(detection)
                self.kalman.update(np.array(center))
        
    def _get_bbox_center(self, detection: Dict[str, Any]) -> Tuple[float, float]:
        """Get center point of bounding box.
        
        Args:
            detection: Detection dict with x_min, y_min, x_max, y_max
            
        Returns:
            (cx, cy) center coordinates
        """
        cx = (detection['x_min'] + detection['x_max']) / 2
        cy = (detection['y_min'] + detection['y_max']) / 2
        return cx, cy
    
    def _get_bbox_size(self, detection: Dict[str, Any]) -> Tuple[float, float]:
        """Get width and height of bounding box.
        
        Args:
            detection: Detection dict with x_min, y_min, x_max, y_max
            
        Returns:
            (width, height) tuple
        """
        w = detection['x_max'] - detection['x_min']
        h = detection['y_max'] - detection['y_min']
        return w, h
    
    def _linear_interpolate_bbox(
        self,
        detection1: Dict[str, Any],
        detection2: Dict[str, Any],
        t: float
    ) -> Dict[str, Any]:
        """Linearly interpolate bounding box between two detections.
        
        Args:
            detection1: First detection (at t=0)
            detection2: Second detection (at t=1)
            t: Interpolation parameter (0 to 1)
            
        Returns:
            Interpolated detection dict
        """
        # Interpolate all bbox coordinates
        x_min = detection1['x_min'] + t * (detection2['x_min'] - detection1['x_min'])
        y_min = detection1['y_min'] + t * (detection2['y_min'] - detection1['y_min'])
        x_max = detection1['x_max'] + t * (detection2['x_max'] - detection1['x_max'])
        y_max = detection1['y_max'] + t * (detection2['y_max'] - detection1['y_max'])
        
        return {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'confidence': self.prediction_confidence,
            'detection_type': 'predicted',
            'class_name': detection1.get('class_name', 'ball'),
            'class_id': detection1.get('class_id', None),
        }
    
    def _kalman_predict_bbox(
        self,
        last_detection: Dict[str, Any],
        steps: int
    ) -> Dict[str, Any]:
        """Predict bounding box using Kalman filter.
        
        Args:
            last_detection: Last known detection (for bbox size reference)
            steps: Number of prediction steps from last detection
            
        Returns:
            Predicted detection dict
        """
        if self.kalman is None:
            raise RuntimeError("Kalman filter not initialized")
        
        # Run prediction steps
        for _ in range(steps):
            self.kalman.predict()
        
        # Get predicted center position
        pred_cx, pred_cy = self.kalman.get_position()
        
        # Use last known bbox size
        w, h = self._get_bbox_size(last_detection)
        
        return {
            'x_min': pred_cx - w / 2,
            'y_min': pred_cy - h / 2,
            'x_max': pred_cx + w / 2,
            'y_max': pred_cy + h / 2,
            'confidence': self.prediction_confidence,
            'detection_type': 'predicted',
            'class_name': last_detection.get('class_name', 'ball'),
            'class_id': last_detection.get('class_id', None),
        }
    
    def fill_gaps(self):
        """Fill detection gaps with motion predictions.
        
        Identifies consecutive frames without detections and fills them using:
        - Linear interpolation for gaps of 1-3 frames
        - Kalman filtering for longer gaps (if use_kalman=True)
        
        This modifies the internal detection history.
        """
        if len(self.detections) < 2:
            return  # Need at least 2 detections to interpolate
        
        # Get sorted list of frames with detections
        detected_frames = sorted(self.detections.keys())
        
        # Find and fill gaps
        for i in range(len(detected_frames) - 1):
            frame1 = detected_frames[i]
            frame2 = detected_frames[i + 1]
            gap_size = frame2 - frame1 - 1
            
            if gap_size == 0 or gap_size > self.max_gap_frames:
                continue  # No gap or gap too large
            
            detection1 = self.detections[frame1]
            detection2 = self.detections[frame2]
            
            # Fill gap based on size and configuration
            if gap_size <= 3 or not self.use_kalman:
                # Linear interpolation for short gaps
                for j in range(1, gap_size + 1):
                    frame_idx = frame1 + j
                    t = j / (gap_size + 1)  # Interpolation parameter
                    pred_detection = self._linear_interpolate_bbox(detection1, detection2, t)
                    self.detections[frame_idx] = pred_detection
                    self.predicted_frames.add(frame_idx)
            
            else:
                # Kalman prediction for longer gaps
                # Note: This uses forward prediction only; could be improved
                # with forward-backward smoothing
                for j in range(1, gap_size + 1):
                    frame_idx = frame1 + j
                    pred_detection = self._kalman_predict_bbox(detection1, j)
                    self.detections[frame_idx] = pred_detection
                    self.predicted_frames.add(frame_idx)
    
    def get_all_detections(self, include_frame_index: bool = True) -> List[Dict[str, Any]]:
        """Get all detections (actual + predicted) sorted by frame.
        
        Args:
            include_frame_index: Whether to include 'frame_index' in each detection dict
            
        Returns:
            List of detection dicts sorted by frame_index
        """
        result = []
        for frame_idx in sorted(self.detections.keys()):
            det = self.detections[frame_idx].copy()
            if include_frame_index:
                det['frame_index'] = frame_idx
            result.append(det)
        return result
    
    def get_detection(self, frame_index: int) -> Optional[Dict[str, Any]]:
        """Get detection for a specific frame.
        
        Args:
            frame_index: Frame number
            
        Returns:
            Detection dict or None if no detection (actual or predicted) exists
        """
        return self.detections.get(frame_index)
    
    def is_predicted(self, frame_index: int) -> bool:
        """Check if detection at frame is predicted.
        
        Args:
            frame_index: Frame number
            
        Returns:
            True if detection is predicted, False if actual detection
        """
        return frame_index in self.predicted_frames


def fill_detection_gaps(
    detections: List[Dict[str, Any]],
    max_gap_frames: int = 5,
    prediction_confidence: float = 0.3,
    use_kalman: bool = False,
    frame_index_key: str = 'frame_index'
) -> List[Dict[str, Any]]:
    """Fill gaps in detection list with motion predictions.
    
    This is a convenience function that wraps BallTracker for simple use cases.
    Takes a list of detection dicts, identifies gaps, and returns a new list
    with predicted detections filled in.
    
    Args:
        detections: List of detection dicts, each with frame_index and bbox coordinates
        max_gap_frames: Maximum gap size to fill
        prediction_confidence: Confidence score for predictions
        use_kalman: Use Kalman filtering for gaps > 3 frames
        frame_index_key: Key name for frame index in detection dicts
    
    Returns:
        New list of detections with gaps filled (sorted by frame_index)
        
    Example:
        ```python
        # Load detections from CSV
        detections = [
            {'frame_index': 0, 'x_min': 100, 'y_min': 200, 'x_max': 150, 'y_max': 250, 'confidence': 0.9},
            {'frame_index': 1, 'x_min': 105, 'y_min': 205, 'x_max': 155, 'y_max': 255, 'confidence': 0.85},
            # Frame 2 missing (gap)
            {'frame_index': 3, 'x_min': 115, 'y_min': 215, 'x_max': 165, 'y_max': 265, 'confidence': 0.88},
        ]
        
        # Fill gaps
        filled = fill_detection_gaps(detections, max_gap_frames=3)
        
        # Now filled[2] contains predicted detection for frame 2
        ```
    """
    if len(detections) == 0:
        return []
    
    # Create tracker
    tracker = BallTracker(
        max_gap_frames=max_gap_frames,
        prediction_confidence=prediction_confidence,
        use_kalman=use_kalman
    )
    
    # Feed detections to tracker
    for det in detections:
        frame_idx = det.get(frame_index_key)
        if frame_idx is None:
            continue
        tracker.update(frame_idx, det)
    
    # Fill gaps
    tracker.fill_gaps()
    
    # Return all detections
    return tracker.get_all_detections(include_frame_index=True)


def filter_ball_detections(
    detections: List[Dict[str, Any]],
    class_name: str = 'sports ball'
) -> List[Dict[str, Any]]:
    """Filter detections to keep only ball detections.
    
    Useful for preprocessing mixed detection results before gap filling.
    
    Args:
        detections: List of all detections
        class_name: Class name to keep (default: 'sports ball')
        
    Returns:
        Filtered list containing only ball detections
    """
    return [d for d in detections if d.get('class_name', '').lower() == class_name.lower()]


def group_detections_by_frame(
    detections: List[Dict[str, Any]],
    frame_index_key: str = 'frame_index'
) -> Dict[int, List[Dict[str, Any]]]:
    """Group detections by frame index.
    
    Args:
        detections: List of detection dicts
        frame_index_key: Key name for frame index
        
    Returns:
        Dict mapping frame_index -> list of detections in that frame
    """
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for det in detections:
        frame_idx = det.get(frame_index_key)
        if frame_idx is None:
            continue
        if frame_idx not in grouped:
            grouped[frame_idx] = []
        grouped[frame_idx].append(det)
    return grouped
