"""Coordinate system (image-space, pixels):
- Origin at top-left.
- +x to the right, +y down.
- Gravity acts in +y.
- Delivery direction (bowler -> batsman) is +x by convention.
"""

import cv2
import numpy as np

class BallKalmanFilter:
    """
    Constant-acceleration Kalman Filter for cricket ball tracking.
    State vector: [x, y, vx, vy, ax, ay]
    Measurement: [x, y]
    """

    def __init__(self, fps: float = 30.0, gravity: float = 0.5):
        """Create a 6-D Kalman filter.
        Args:
            fps: Frames per second of the video (used to compute dt).
            gravity: Positive down-wards acceleration (pixels/frame²) applied to ay.
        """
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self.dt = 1.0 / fps
        self.gravity = gravity
        # 6 state variables, 2 measurements
        self.kf = cv2.KalmanFilter(6, 2)
        # Transition matrix A (constant acceleration model)
        dt = 1.0 # Using frames as time unit for simplicity in CV coordinates
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0],   # x
            [0, 1, 0, dt, 0, 0.5*dt*dt],   # y
            [0, 0, 1, 0, dt, 0],           # vx
            [0, 0, 0, 1, 0, dt],           # vy
            [0, 0, 0, 0, 1, 0],            # ax
            [0, 0, 0, 0, 0, 1]             # ay
        ], dtype=np.float32)
        
        # Measurement matrix H - we observe x and y
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance (Q)
        q = np.eye(6, dtype=np.float32) * 0.03
        q[4:, 4:] *= 5.0 # ax, ay higher uncertainty
        self.kf.processNoiseCov = q
        
        # Measurement noise covariance (R)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        # Error covariance (P) - initial uncertainty
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1.0
        
        self.orig_process_noise = self.kf.processNoiseCov.copy()
        self.orig_measurement_noise = self.kf.measurementNoiseCov.copy()

    def initialize(self, x, y, vx=0, vy=0, ax=0, ay=None):
        """Set initial state."""
        if ay is None:
            ay = self.gravity
        self.kf.statePost = np.array([[x], [y], [vx], [vy], [ax], [ay]], dtype=np.float32)
        
    def predict(self):
        """Predict next state and apply gravity bias."""
        # Inject gravity into acceleration state ay
        self.kf.statePost[5][0] = self.gravity
        prediction = self.kf.predict()
        # Keep statePost aligned with prediction when no measurement is available.
        self.kf.statePost = prediction.copy()
        return float(prediction[0][0]), float(prediction[1][0])
    
    def update(self, x, y):
        """Update with measurement."""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)
        
    def get_state(self):
        """Return [x, y, vx, vy] for backward compatibility."""
        s = self.kf.statePost
        return float(s[0][0]), float(s[1][0]), float(s[2][0]), float(s[3][0])

    def get_full_state(self):
        """Return the complete 6-D state (x, y, vx, vy, ax, ay)."""
        s = self.kf.statePost
        return tuple(float(v[0]) for v in s)

    def relax_constraints(self):
        """Relax constraints during bounce or occlusion."""
        self.kf.processNoiseCov = self.orig_process_noise * 50.0 
        self.kf.measurementNoiseCov = self.orig_measurement_noise * 5.0
        
    def reset_constraints(self):
        """Restore original physics constraints."""
        self.kf.processNoiseCov = self.orig_process_noise
        self.kf.measurementNoiseCov = self.orig_measurement_noise

    def set_measurement_noise_scale(self, scale: float):
        """Scale measurement noise to down-weight uncertain measurements."""
        if scale <= 0:
            scale = 1.0
        self.kf.measurementNoiseCov = self.orig_measurement_noise * float(scale)

    def set_process_noise_scale(self, scale: float):
        """Scale process noise to allow more model flexibility."""
        if scale <= 0:
            scale = 1.0
        self.kf.processNoiseCov = self.orig_process_noise * float(scale)

    def set_state(self, x=None, y=None, vx=None, vy=None, ax=None, ay=None):
        """Update components of the state vector in-place."""
        if x is not None:
            self.kf.statePost[0][0] = float(x)
        if y is not None:
            self.kf.statePost[1][0] = float(y)
        if vx is not None:
            self.kf.statePost[2][0] = float(vx)
        if vy is not None:
            self.kf.statePost[3][0] = float(vy)
        if ax is not None:
            self.kf.statePost[4][0] = float(ax)
        if ay is not None:
            self.kf.statePost[5][0] = float(ay)

    def force_velocity_flip(self):
        """Force vertical velocity to invert (bounce)."""
        vy = self.kf.statePost[3][0]
        if vy > 0: # If moving down
            self.kf.statePost[3][0] = -vy * 0.8 
            self.kf.errorCovPost[3][3] = 100.0

    def set_gravity_bias(self, g_per_frame=0.5):
        """Update the gravity bias."""
        self.gravity = g_per_frame
