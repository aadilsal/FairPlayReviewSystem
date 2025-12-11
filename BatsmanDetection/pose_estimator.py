# pose_estimator.py
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 pose model (pretrained on COCO keypoints)
pose_model = YOLO("weights/yolov8s-pose.pt")

def estimate_pose(frame, bbox=None, conf_threshold=0.5):
    """
    Run pose estimation, optionally constrained to a given bounding box (bbox).
    
    Args:
        frame (np.array): The full input frame.
        bbox (list/tuple, optional): [x, y, w, h] of the Batsman area.
        conf_threshold (float): Confidence threshold for pose detection.
        
    Returns:
        frame (np.array): The UNTOUCHED original frame.
        keypoints_all (list): List of keypoints for all detected persons (usually 1).
                              Each item is an array of shape (17, 2) or (17, 3).
    """
    
    x_offset, y_offset = 0, 0
    
    # 1. CROP FRAME IF BBOX IS PROVIDED
    if bbox is not None and len(bbox) == 4:
        # Unpack and ensure integers
        x, y, w, h = map(int, bbox)
        
        # Add a small buffer around the box for better detection performance
        buffer_x = w // 4
        buffer_y = h // 4
        
        # Clamp coordinates to frame boundaries
        h_frame, w_frame = frame.shape[:2]
        
        x_start = max(0, x - buffer_x)
        y_start = max(0, y - buffer_y)
        x_end = min(w_frame, x + w + buffer_x)
        y_end = min(h_frame, y + h + buffer_y)
        
        # Update offsets (needed to translate keypoints back)
        x_offset = x_start
        y_offset = y_start
        
        # Crop the frame
        input_frame = frame[y_start:y_end, x_start:x_end]
    else:
        # Use full frame if no bbox is provided (fallback)
        input_frame = frame

    # 2. RUN POSE ESTIMATION on the cropped (or full) area
    results = pose_model.predict(input_frame, conf=conf_threshold, verbose=False)

    keypoints_all = []

    for result in results:
        # Check if any person was detected
        if result.keypoints.xy.shape[0] == 0:
            continue
            
        for person_keypoints in result.keypoints.xy:
            person_keypoints = person_keypoints.cpu().numpy()
            
            # 3. TRANSLATE KEYPOINTS BACK TO FULL FRAME COORDINATES
            if x_offset != 0 or y_offset != 0:
                person_keypoints[:, 0] += x_offset # Adjust X coordinate
                person_keypoints[:, 1] += y_offset # Adjust Y coordinate

            keypoints_all.append(person_keypoints)

    # 4. RETURN WITHOUT DRAWING
    return frame, keypoints_all