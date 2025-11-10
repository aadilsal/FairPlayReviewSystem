"""Simple tracking utilities for ball trajectories (lightweight)."""
import numpy as np
from typing import List, Dict, Optional


class SimpleKalmanTracker:
    """Very small tracker using constant velocity approximation.

    This is intentionally simple; it provides predicted positions and basic
    track bookkeeping so the detector can recover short-term misses.
    """
    def __init__(self, max_age: int = 30):
        self.max_age = max_age
        self.tracks = []
        self.next_id = 0

    def update(self, detections: List[Dict]) -> List[Dict]:
        # If no tracks, initialize with first detection
        if len(self.tracks) == 0 and len(detections) > 0:
            det = detections[0]
            bbox = det.get('bbox') if isinstance(det, dict) else None
            if bbox is None and isinstance(det, (list, tuple)):
                x, y, w, h = det[0:4]
                cx = x + w / 2
                cy = y + h / 2
            else:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2

            self.tracks.append({
                'id': self.next_id,
                'position': np.array([cx, cy]),
                'velocity': np.array([0.0, 0.0]),
                'age': 0,
                'hits': 1,
            })
            self.next_id += 1
            if isinstance(detections[0], dict):
                detections[0]['track_id'] = 0
            else:
                # wrap primitive detection into dict
                detections[0] = {'bbox': detections[0][0:4], 'track_id': 0}

        # Age tracks and remove old
        for t in self.tracks:
            t['age'] += 1
        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]

        return detections

    def predict_position(self) -> Optional[np.ndarray]:
        if len(self.tracks) == 0:
            return None
        t = self.tracks[0]
        return t['position'] + t['velocity']
