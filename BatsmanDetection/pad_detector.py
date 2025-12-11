# file: pad_detector.py
import numpy as np

class PadDetector:
    def __init__(self, width_ratio=0.5, foot_height_ratio=0.25):
        """
        Initializes the Pad and Foot detector.
        """
        self.width_ratio = width_ratio
        self.foot_height_ratio = foot_height_ratio

        # COCO Keypoint Indices
        self.LEFT_KNEE = 13
        self.RIGHT_KNEE = 14
        self.LEFT_ANKLE = 15
        self.RIGHT_ANKLE = 16

    def calculate_box(self, knee, ankle, label_prefix):
        """
        Generates bounding boxes for the Pad and Foot based on knee and ankle coordinates.
        Handles both [x, y] and [x, y, conf] formats.
        """
        detections = []
        
        # --- SAFE UNPACKING START ---
        # Handle Knee
        if len(knee) >= 3:
            kx, ky, k_conf = knee[:3]
        else:
            kx, ky = knee[:2]
            # If data is only [x, y], assume valid if coordinates are non-zero
            k_conf = 1.0 if (kx > 0 and ky > 0) else 0.0

        # Handle Ankle
        if len(ankle) >= 3:
            ax, ay, a_conf = ankle[:3]
        else:
            ax, ay = ankle[:2]
            # If data is only [x, y], assume valid if coordinates are non-zero
            a_conf = 1.0 if (ax > 0 and ay > 0) else 0.0
        # --- SAFE UNPACKING END ---

        # Threshold to ensure we actually have valid keypoints
        # (Using a low threshold because inferred conf might be 1.0)
        if k_conf < 0.1 or a_conf < 0.1:
            return detections
            
        # Also check for 0,0 coordinates explicitly (common in pose output for "not found")
        if kx == 0 or ky == 0 or ax == 0 or ay == 0:
            return detections

        # 1. Calculate Leg Height
        leg_height = abs(ay - ky)
        
        # If leg is compressed/hidden or noise (too small), skip
        if leg_height < 10: 
            return detections

        # 2. Calculate Dynamic Widths
        box_width = leg_height * self.width_ratio
        
        # --- PAD BOX CALCULATION ---
        center_x = (kx + ax) / 2
        pad_x1 = center_x - (box_width / 2)
        pad_y1 = ky
        pad_h = leg_height
        
        detections.append({
            "label": f"{label_prefix}_Pad",
            "conf": (k_conf + a_conf) / 2, 
            "box": [int(pad_x1), int(pad_y1), int(box_width), int(pad_h)] 
        })

        # --- FOOT BOX CALCULATION ---
        foot_h = leg_height * self.foot_height_ratio
        foot_x1 = ax - (box_width / 2)
        foot_y1 = ay - (foot_h / 2)
        
        detections.append({
            "label": f"{label_prefix}_Foot",
            "conf": a_conf,
            "box": [int(foot_x1), int(foot_y1), int(box_width), int(foot_h)]
        })

        return detections

    def detect(self, keypoints_list):
        """
        Process pose keypoints to find pads and feet.
        
        Args:
            keypoints_list: A list of keypoint arrays. 
        """
        pad_detections = []
        
        # Handle empty input
        if not keypoints_list:
            return pad_detections
        
        for person_kps in keypoints_list:
            # Ensure it's numpy or list accessible
            kps = np.array(person_kps) if not isinstance(person_kps, np.ndarray) else person_kps

            # Basic check for COCO format length (17 points)
            if len(kps) < 17:
                continue

            # Get Left Leg points
            l_knee = kps[self.LEFT_KNEE]
            l_ankle = kps[self.LEFT_ANKLE]
            
            # Get Right Leg points
            r_knee = kps[self.RIGHT_KNEE]
            r_ankle = kps[self.RIGHT_ANKLE]

            # Calculate Left Side
            pad_detections.extend(self.calculate_box(l_knee, l_ankle, "Left"))
            
            # Calculate Right Side
            pad_detections.extend(self.calculate_box(r_knee, r_ankle, "Right"))

        return pad_detections

# ---------------------------------------------------------
# MODULE CALLER FUNCTION
# ---------------------------------------------------------
def detect_pads(frame, keypoints_list, conf=0.3):
    """
    Main caller function integrated with detection_pipeline.py.
    
    Args:
        frame (np.array): The image frame.
        keypoints_list (list): The list of pose keypoints (det_pose).
        conf (float): Confidence threshold.
    """
    detector = PadDetector()
    
    # Generate all potential pad/foot boxes from the keypoints
    raw_detections = detector.detect(keypoints_list)
    
    # Filter detections based on the provided confidence threshold
    final_pads = [p for p in raw_detections if p['conf'] >= conf]

    return frame, final_pads