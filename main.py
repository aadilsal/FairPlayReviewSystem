import argparse
from frame_extractor import extract_video_frames
from detection_pipeline import process_frames_pipeline
from video_utils import frames_to_video_with_custom_path

def main():
    parser = argparse.ArgumentParser(description='FairPlayReviewSystem')
    parser.add_argument('--input', '-i', required=True, help='Path to input video file')
    parser.add_argument('--output', '-o', default='outputs', help='Output directory')
    parser.add_argument('--fps', type=int, default=30, help='FPS for frame extraction and output video')
    args = parser.parse_args()

    frame_paths, frames_dir = extract_video_frames(args.input, args.output, args.fps)
    process_frames_pipeline(frame_paths)
    output_video_path = frames_to_video_with_custom_path(args.input, frames_dir, args.fps, args.output)
    print(f"Output video saved to: {output_video_path}")

if __name__ == "__main__":
    main()