import os
import cv2
from pathlib import Path

def frames_to_video_with_custom_path(frames_dir: str, out_path: str, fps: int = 30):
    """
    Read all JPG/PNG files in frames_dir, sort them, and write a video to out_path.
    """
    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not frame_files:
        raise RuntimeError(f"No image frames found in {frames_dir}")

    # read first frame to get size
    first = cv2.imread(str(frames_dir / frame_files[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frames_dir / frame_files[0]}")
    h, w = first.shape[:2]

    # ensure output dir exists
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_path}")

    for fname in frame_files:
        img_path = frames_dir / fname
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Skipping unreadable frame: {img_path}")
            continue
        # resize if needed to match first frame
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)

    writer.release()
    print(f"[INFO] Video written: {out_path}")