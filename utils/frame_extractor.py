import os
from pathlib import Path
import cv2

def extract_video_frames(video_path: str, output_dir: str, target_fps: int = 30):
    """
    Extract frames from `video_path` into `output_dir` at approx `target_fps`.
    - If frames already exist in output_dir, returns the existing list.
    - Avoids ZeroDivisionError when video fps is 0 or target_fps > source fps.
    - Returns a sorted list of full frame file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # if frames already present, reuse them
    existing = sorted([str(out_dir / f) for f in os.listdir(out_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if existing:
        print(f"Frames already exist for {out_dir.name}, skipping extraction.")
        return existing

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 0:
        # fallback when FPS cannot be read
        src_fps = 30.0

    if target_fps <= 0:
        target_fps = int(src_fps)

    # compute frame step and avoid zero division
    step = max(1, int(round(src_fps / float(target_fps))))

    frame_paths = []
    read_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if read_idx % step == 0:
            fname = out_dir / f"frame_{saved_idx:06d}.jpg"
            cv2.imwrite(str(fname), frame)
            frame_paths.append(str(fname))
            saved_idx += 1
        read_idx += 1

    cap.release()
    return frame_paths