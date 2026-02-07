import sys
import os
import numpy as np
import cv2

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BallDetection')))

from ball_detector import HybridBallTracker
from kalman_filter import BallKalmanFilter

class MockDetector:
    def __init__(self):
        self.result = []
    def detect(self, frame, **kwargs):
        return self.result

def test_parabolic_trajectory_with_occlusion():
    """
    Test that the tracker maintains a parabolic trajectory even when YOLO fails.
    """
    tracker = HybridBallTracker()
    detector = MockDetector()
    
    # Generate synthetic parabolic trajectory
    xs = np.linspace(100, 540, 60)
    ys = 0.005 * (xs - 320)**2 + 200
    
    successful_frames = 0
    
    for i, (x, y) in enumerate(zip(xs, ys)):
        # Create a new unique frame
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        # Simulate occlusion: don't draw the ball between frame 20 and 40
        if not (20 <= i <= 40):
            cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), -1)

        # Simulate YOLO dropout between frame 20 and 40
        if 20 <= i <= 40:
            detector.result = []
        else:
            detector.result = [[x-5, y-5, 10, 10, 0.9, 0]]
            
        print(f"--- TEST Frame {i} START ---")
        try:
            ball_info = tracker.process_frame(frame, i, detector)
        except Exception as e:
            print(f"!!! CRASH at Frame {i}: {e}")
            import traceback
            traceback.print_exc()
            break
        print(f"--- TEST Frame {i} END ---")
        
        if ball_info:
            successful_frames += 1
            bx, by, bw, bh = ball_info['box']
            cx, cy = bx + bw/2, by + bh/2
            dist = np.hypot(cx - x, cy - y)
            print(f"Frame {i}: Pos=({cx:.1f}, {cy:.1f}), GT=({x:.1f}, {y:.1f}), Dist={dist:.2f}, Source={ball_info['source']}")
            
            if 20 <= i <= 40:
                if ball_info['source'] not in ["prediction_only", "guided_recovery", "kalman_prediction", "kalman_coast"]:
                    print(f"!!! SOURCE ERROR at frame {i}: {ball_info['source']}")
                assert ball_info['source'] in ["prediction_only", "guided_recovery", "kalman_prediction", "kalman_coast"]
        else:
            print(f"Frame {i}: LOST")
            
    print(f"Total frames: {len(xs)}, Successful: {successful_frames}")
    # We expect 100% success because Kalman should coast through occlusion
    assert successful_frames == len(xs)

if __name__ == "__main__":
    try:
        test_parabolic_trajectory_with_occlusion()
        print("\n[SUCCESS] Parabolic trajectory test passed!")
    except Exception as e:
        print(f"\n[FAILURE] Test failed: {e}")
        import traceback
        traceback.print_exc()
