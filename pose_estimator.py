# pose_estimator.py
import cv2
from ultralytics import YOLO

# Load YOLOv8 pose model (pretrained on COCO keypoints)
pose_model = YOLO("yolov8s-pose.pt")  # you can also try yolov8n-pose.pt for faster inference

def estimate_pose(frame, conf_threshold=0.5):
    """
    Run pose estimation on a frame and draw skeleton.
    Returns frame with skeleton drawn + list of keypoints for each person.
    """
    results = pose_model.predict(frame, conf=conf_threshold, verbose=False)

    keypoints_all = []

    for result in results:
        for person_keypoints in result.keypoints.xy:  # list of keypoints for each person
            person_keypoints = person_keypoints.cpu().numpy()
            keypoints_all.append(person_keypoints)

            # Draw keypoints
            for (x, y) in person_keypoints:
                if x > 0 and y > 0:  # valid keypoint
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

            # Draw skeleton connections (COCO 17-keypoint format)
            skeleton_pairs = [
                (5, 7), (7, 9),    # left arm
                (6, 8), (8, 10),   # right arm
                (11, 13), (13, 15),# left leg
                (12, 14), (14, 16),# right leg
                (5, 6),            # shoulders
                (11, 12),          # hips
                (5, 11), (6, 12)   # torso
            ]

            for (i, j) in skeleton_pairs:
                if person_keypoints[i][0] > 0 and person_keypoints[j][0] > 0:
                    pt1 = (int(person_keypoints[i][0]), int(person_keypoints[i][1]))
                    pt2 = (int(person_keypoints[j][0]), int(person_keypoints[j][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    return frame, keypoints_all
