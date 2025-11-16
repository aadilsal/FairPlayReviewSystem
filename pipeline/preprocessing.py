import os
import cv2


def extract_video_frames(video_path, output_dir, target_fps=30):
    """Extract frames from video and save as JPEGs.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where frames will be saved. Frames are placed in
            output_dir/frames/<video_name>.
        target_fps (int): Approximate number of frames to extract per second.

    Returns:
        tuple: (list_of_frame_paths, frames_dir)
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(output_dir, "frames", video_name)
    os.makedirs(frames_dir, exist_ok=True)
    frame_paths = []

    # Check if frames already exist
    existing_frames = [f for f in os.listdir(frames_dir) if f.lower().endswith('.jpg')]
    if existing_frames:
        print(f"Frames already exist for {video_name}, skipping extraction.")
        return [os.path.join(frames_dir, f) for f in sorted(existing_frames)], frames_dir

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    # Prevent division by zero and ensure integer interval
    frame_interval = int(round(original_fps / target_fps)) if original_fps > target_fps and target_fps > 0 else 1
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_path = os.path.join(frames_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved += 1
        count += 1
    cap.release()
    print(f"Extracted {len(frame_paths)} frames to {frames_dir}")
    return frame_paths, frames_dir
