import argparse
import os

from frame_extractor import extract_video_frames
from detection_pipeline import process_frames_pipeline
from video_utils import frames_to_video_with_custom_path


def process_single_video(input_path, output_dir, fps):
    frame_paths, frames_dir = extract_video_frames(input_path, output_dir, fps)
    process_frames_pipeline(frame_paths)
    output_video_path = frames_to_video_with_custom_path(input_path, frames_dir, fps, output_dir)
    print(f"Output video saved to: {output_video_path}")


def process_folder(folder_path, output_dir, fps):
    videos = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
              if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not videos:
        print(f"No video files found in {folder_path}")
        return
    for vid in videos:
        print(f"\n=== Processing {vid} ===")
        process_single_video(vid, output_dir, fps)


def main():
    parser = argparse.ArgumentParser(description='Run FairPlayReviewSystem pipeline')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to input video file or folder containing videos')
    parser.add_argument('--output', '-o', default='outputs', help='Output directory')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS for extraction and output video')
    args = parser.parse_args()

    if os.path.isdir(args.input):
        process_folder(args.input, args.output, args.fps)
    else:
        process_single_video(args.input, args.output, args.fps)


if __name__ == '__main__':
    main()