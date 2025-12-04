import os
from pathlib import Path
import cv2
import argparse


def extract_video_frames(video_path, output_dir, target_fps=30):
    """Extract frames from video and save as JPEGs.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where frames will be saved directly (no subdirs created).
        target_fps (int): Approximate number of frames to extract per second.

    Returns:
        list: list_of_frame_paths
    """
    frames_dir = output_dir
    os.makedirs(frames_dir, exist_ok=True)
    frame_paths = []

    # Check if frames already exist
    existing_frames = [f for f in os.listdir(frames_dir) if f.lower().endswith('.jpg')]
    if existing_frames:
        print(f"Frames already exist for {Path(frames_dir).name}, skipping extraction.")
        return [os.path.join(frames_dir, f) for f in sorted(existing_frames)]

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

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    # Prevent division by zero and ensure integer interval
    frame_interval = int(round(original_fps / target_fps)) if original_fps > target_fps and target_fps > 0 else 1
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if read_idx % step == 0:
            fname = os.path.join(frames_dir, f"frame_{saved_idx:06d}.jpg")
            cv2.imwrite(str(fname), frame)
            frame_paths.append(str(fname))
            saved_idx += 1
        read_idx += 1

    cap.release()
    return frame_paths


def _parse_args():
    p = argparse.ArgumentParser(description="Extract frames from a video file")
    p.add_argument("-i", "--input", required=True, help="Path to input video file")
    p.add_argument("-o", "--output", default="outputs", help="Output directory (default: outputs)")
    p.add_argument("-f", "--fps", type=int, default=30, help="Target frames per second to extract (default: 30)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    frames, dirpath = extract_video_frames(args.input, args.output, args.fps)
    print(f"Done. Frames saved to: {dirpath}")