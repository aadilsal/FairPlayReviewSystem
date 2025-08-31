import cv2
import os

def extract_frames(video_path, output_folder="outputs/frame", target_fps=30):
    import cv2, os
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(original_fps / target_fps)) if original_fps > target_fps else 1

    frame_paths = []
    count = 0
    saved_count = 0
    os.makedirs(output_folder, exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1
        count += 1
    cap.release()
    return frame_paths