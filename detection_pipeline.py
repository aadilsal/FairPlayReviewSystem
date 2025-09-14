import cv2
from ball_detector import detect_ball_on_frame
from person_detector import detect_persons
from pose_estimator import estimate_pose
import os

def process_frames_pipeline(frame_paths):
    for frame_path in frame_paths:
        pose_marker = frame_path + ".pose"
        if os.path.exists(pose_marker):
            print(f"[INFO] All detections already done for {frame_path}, skipping.")
            continue

        person_marker = frame_path + ".person"
        ball_marker = frame_path + ".ball"

        frame = cv2.imread(frame_path)

        # Ball detection (YOLO, fallback to color)
        if not os.path.exists(ball_marker):
            frame_with_ball, ball_detected = detect_ball_on_frame(frame)
            cv2.imwrite(frame_path, frame_with_ball)
            with open(ball_marker, "w") as f:
                f.write("ball detected")
        else:
            frame_with_ball = cv2.imread(frame_path)

        # Person detection
        if not os.path.exists(person_marker):
            frame_with_persons, _ = detect_persons(frame_with_ball)
            cv2.imwrite(frame_path, frame_with_persons)
            with open(person_marker, "w") as f:
                f.write("person detected")
        else:
            frame_with_persons = cv2.imread(frame_path)

        # Pose estimation
        frame_with_pose, _ = estimate_pose(frame_with_persons)
        cv2.imwrite(frame_path, frame_with_pose)
        with open(pose_marker, "w") as f:
            f.write("pose estimated")

        # --- Display the processed frame ---
        cv2.imshow("Processed Frame", frame_with_pose)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()