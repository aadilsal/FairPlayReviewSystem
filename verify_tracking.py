import cv2
import sys
import os

# Add BallDetection to path since we are in root
sys.path.append(os.path.join(os.getcwd(), 'BallDetection'))

from BallDetection.ball_detector import detect_ball_on_frame, get_hybrid_tracker

def verify_tracking():
    video_path = r"test_videos/lbw.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    # Track sources
    sources = []
    
    print("Starting verification loop...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run detection
        # We assume weights are in "weights/ball-yolov8s.pt" relative to root or we need to pass absolute
        # Let's hope the default works or we might need to adjust
        
        frame, ball_info = detect_ball_on_frame(frame, frame_idx=frame_idx, debug=False)
        
        if ball_info:
            source = ball_info['source']
            sources.append(source)
            print(f"Frame {frame_idx}: {source} - Pos: {ball_info['box']}")
        else:
            sources.append("None")
            print(f"Frame {frame_idx}: No ball")
            
        frame_idx += 1
        if frame_idx > 100: # Test first 100 frames
            break
            
    cap.release()
    
    # Analysis
    yolo_count = sum(1 for s in sources if 'yolo' in s and 'correction' not in s)
    tracking_count = sum(1 for s in sources if 'optical_flow' in s or 'csrt' in s or 'kalman' in s)
    correction_count = sum(1 for s in sources if 'correction' in s)
    
    print("\n--- Summary ---")
    print(f"Total Frames: {frame_idx}")
    print(f"YOLO Detections (Bootstrap): {yolo_count}")
    print(f"Tracking Detections (OF/CSRT/KF): {tracking_count}")
    print(f"Corrections: {correction_count}")
    
    if tracking_count > 0:
        print("SUCCESS: System transitioned to tracking mode.")
    else:
        print("FAILURE: System relied entirely on YOLO or detected nothing.")

if __name__ == "__main__":
    verify_tracking()
