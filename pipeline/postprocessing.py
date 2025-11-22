import cv2
import os

def frames_to_video_with_custom_path(input_video_path, frames_dir, fps=30, output_root="outputs"):
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_video_folder = os.path.join(output_root, "frames", video_name)
    os.makedirs(output_video_folder, exist_ok=True)
    output_video_path = os.path.join(output_video_folder, f"output_{video_name}.mp4")

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    if not frame_files:
        raise ValueError("No frames found to combine into video.")
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for fname in frame_files:
        frame = cv2.imread(os.path.join(frames_dir, fname))
        out.write(frame)
    out.release()
    return output_video_path
