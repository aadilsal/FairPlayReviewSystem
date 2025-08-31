import cv2
import os

def extract_frames(video_path, output_folder="outputs/frames", target_fps=30):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(original_fps / target_fps))

    frame_count = 0
    saved_count = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save only every `frame_interval` frames
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            saved_count += 1

        frame_count += 1

    cap.release()
    return frames
