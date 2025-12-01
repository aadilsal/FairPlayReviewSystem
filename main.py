import argparse
import os
from pathlib import Path
from frame_extractor import extract_video_frames
from detection_pipeline import process_frames_pipeline
from video_utils import frames_to_video_with_custom_path

def main():
    parser = argparse.ArgumentParser(description='FairPlayReviewSystem - Batsman Detection & Tracking')
    parser.add_argument('--input', '-i', required=True, help='Path to input video file')
    parser.add_argument('--output', '-o', default='outputs/frames', help='Output directory (default: outputs)')
    parser.add_argument('--fps', type=int, default=30, help='FPS for output video (default: 30)')
    parser.add_argument('--person-conf', type=float, default=0.5, help='Person detection confidence (default: 0.5)')
    parser.add_argument('--bat-conf', type=float, default=0.2, help='Bat detection confidence (default: 0.3)')
    parser.add_argument('--iou-thresh', type=float, default=0.12, help='IoU threshold for bat-person overlap (default: 0.12)')
    parser.add_argument('--consec-frames', type=int, default=3, help='Consecutive frames required to lock batsman (default: 3)')
    #parser.add_argument('--pos-tolerance', type=int, default=50, help='Position tolerance in pixels (default: 50)')

    args = parser.parse_args()

    # validate input file exists
    if not os.path.exists(args.input):
        print(f"[ERROR] Input video file not found: {args.input}")
        return

    # create output directory
    if not os.path.exists(args.output):
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    

    # extract video name (without extension)
    video_name = Path(args.input).stem
    frames_dir = Path(args.output) / video_name
    if not os.path.exists(frames_dir):
        frames_dir.mkdir(parents=True, exist_ok=True)
        
    

    print(f"[INFO] Extracting frames from {args.input}...")
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
    #print(f"  - Position tolerance: {args.pos_tolerance} px")

    # run detection pipeline with arguments
    process_frames_pipeline(
        frame_paths,
        person_conf=args.person_conf,
        bat_conf=args.bat_conf,
        iou_thresh=args.iou_thresh,
        consec_required=args.consec_frames
    )

    print(f"[INFO] Detection pipeline completed.")

    print(f"[INFO] Creating output video...")
    output_video_path = frames_dir / f"{video_name}_output.mp4"
    frames_to_video_with_custom_path(str(frames_dir), str(output_video_path), args.fps)
    print(f"[INFO] Output video saved to {output_video_path}")

if __name__ == "__main__":
    main()