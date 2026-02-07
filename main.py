import argparse
import os
import time
import sys
from pathlib import Path

current_dir = os.getcwd()
sys.path.append(current_dir) 
sys.path.append(os.path.join(current_dir, "BallDetection"))
sys.path.append(os.path.join(current_dir, "BatsmanDetection"))
sys.path.append(os.path.join(current_dir, "WicketDetection"))
sys.path.append(os.path.join(current_dir, "Pipeline"))
sys.path.append(os.path.join(current_dir, "utils"))

try:
    from utils.frame_extractor import extract_video_frames
    from utils.video_utils import frames_to_video_with_custom_path
    from detection_pipeline import process_frames_pipeline 
except ImportError as e:
    print(f"[CRITICAL ERROR] Could not import project modules: {e}")
    print("Ensure you are running the script from the root directory 'FairPlayReviewSystem'.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='FairPlayReviewSystem - Batsman Detection & Tracking')
    parser.add_argument('--input', '-i', required=True, help='Path to input video file')
    parser.add_argument('--output', '-o', default='outputs/frames', help='Base Output directory (default: outputs/frames)')
    parser.add_argument('--fps', type=int, default=30, help='FPS for output video (default: 30)')
    parser.add_argument('--person-conf', type=float, default=0.5, help='Person detection confidence (default: 0.5)')
    parser.add_argument('--bat-conf', type=float, default=0.1, help='Bat detection confidence (default: 0.2)')
    parser.add_argument('--iou-thresh', type=float, default=0.05, help='IoU threshold for bat-person overlap (default: 0.05)')
    parser.add_argument('--consec-frames', type=int, default=3, help='Consecutive frames required to lock batsman (default: 3)')
    parser.add_argument('--wicket-conf', type=float, default=0.25, help='Wicket detection confidence (default: 0.25)')

    args = parser.parse_args()

    # validate input file exists
    if not os.path.exists(args.input):
        print(f"[ERROR] Input video file not found: {args.input}")
        return

    # Extract video name
    video_name_stem = Path(args.input).stem
    
    # Format: outputs/frames/videoName_YYYYMMDD_HHMMSS
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_folder_name = f"{video_name_stem}_{timestamp}"
    
    # The full path to the new unique directory
    frames_dir = Path(args.output) / unique_folder_name
    
    # Create the directory
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Created new output directory: {frames_dir}")
    # ---------------------------------------------------------

    print(f"[INFO] Extracting frames from {args.input}...")
    # Now passing the unique 'frames_dir'
    frame_paths_result = extract_video_frames(args.input, str(frames_dir), args.fps)

    # Always build a deterministic, sorted list of frame files from frames_dir
    frame_files = sorted([p for p in os.listdir(frames_dir) if p.lower().endswith(('.jpg', '.png'))])
    frame_paths = [str(frames_dir / f) for f in frame_files]

    print(f"[INFO] Extracted {len(frame_paths)} frames to {frames_dir}")

    print(f"[INFO] Running detection pipeline...")
    print(f"  - Person confidence: {args.person_conf}")
    print(f"  - Bat confidence: {args.bat_conf}")
    print(f"  - IoU threshold: {args.iou_thresh}")
    print(f"  - Consecutive frames required: {args.consec_frames}")
    print(f"  - Wicket confidence: {args.wicket_conf}")

    # run detection pipeline with arguments
    process_frames_pipeline(
        frame_paths,
        person_conf=args.person_conf,
        bat_conf=args.bat_conf,
        iou_thresh=args.iou_thresh,
        consec_required=args.consec_frames,
        wicket_conf=args.wicket_conf
    )

    print(f"[INFO] Detection pipeline completed.")

    print(f"[INFO] Creating output video...")
    output_video_path = frames_dir / f"{video_name_stem}_output.mp4"
    frames_to_video_with_custom_path(str(frames_dir), str(output_video_path), args.fps)
    print(f"[INFO] Output video saved to {output_video_path}")

if __name__ == "__main__":
    main()
