import argparse
import os
import time
import shutil
import sys
from pathlib import Path
from global_config import GLOBAL_CONFIG

current_dir = os.getcwd()
sys.path.append(current_dir) 
sys.path.append(os.path.join(current_dir, "BallDetection"))
sys.path.append(os.path.join(current_dir, "BatsmanDetection"))
sys.path.append(os.path.join(current_dir, "WicketDetection"))
sys.path.append(os.path.join(current_dir, "Pipeline"))
sys.path.append(os.path.join(current_dir, "utils"))
sys.path.append(os.path.join(current_dir, "LbwDecision"))

try:
    from frame_extractor import extract_video_frames
    from video_utils import frames_to_video_with_custom_path
    from detection_pipeline import process_frames_pipeline 
except ImportError as e:
    print(f"[CRITICAL ERROR] Could not import project modules: {e}")
    print("Ensure you are running the script from the root directory 'FairPlayReviewSystem'.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='FairPlayReviewSystem - Batsman Detection & Tracking')
    parser.add_argument('--input', '-i', required=True, help='Path to input video file')
    parser.add_argument('--output', '-o', default='outputs/frames', help='Base Output directory (default: outputs/frames)')
    parser.add_argument('--fps', type=int, default=GLOBAL_CONFIG['fps'] if 'fps' in GLOBAL_CONFIG else 60, help='FPS for output video (default: 60)')
    parser.add_argument('--person-conf', type=float, default=GLOBAL_CONFIG.get('person_conf'), help='Person detection confidence (default: 0.5)')
    parser.add_argument('--bat-conf', type=float, default=GLOBAL_CONFIG.get('bat_conf'), help='Bat detection confidence (default: 0.1)')
    parser.add_argument('--iou-thresh', type=float, default=GLOBAL_CONFIG.get('iou_thresh'), help='IoU threshold for bat-person overlap (default: 0.05)')
    parser.add_argument('--consec-frames', type=int, default=GLOBAL_CONFIG.get('consec_frames'), help='Consecutive frames required to lock batsman (default: 3)')
    parser.add_argument('--wicket-conf', type=float, default=GLOBAL_CONFIG.get('wicket_conf'), help='Wicket detection confidence (default: 0.25)')
    parser.add_argument('--pad-conf', type=float, default=GLOBAL_CONFIG.get('pad_conf'), help='Pad detection confidence (default: 0.3)')
    parser.add_argument(
        '--disable-dynamic-wicket',
        action='store_true',
        help='Disable per-frame dynamic wicket detection for CLI runs (default: off).',
    )
    parser.add_argument(
        '--enable-dynamic-wicket',
        action='store_true',
        help='Explicitly enable per-frame dynamic wicket detection for CLI runs.',
    )
    args = parser.parse_args()

    if args.disable_dynamic_wicket and args.enable_dynamic_wicket:
        print("[ERROR] Choose only one of --disable-dynamic-wicket or --enable-dynamic-wicket")
        return

    # CLI policy: disable dynamic wicket detection by default to avoid unstable wicket tracks.
    dynamic_wicket_detection = args.enable_dynamic_wicket
    if args.disable_dynamic_wicket:
        dynamic_wicket_detection = False

    # validate input file exists
    if not os.path.exists(args.input):
        print(f"[ERROR] Input video file not found: {args.input}")
        return

    video_name_stem = Path(args.input).stem
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_folder_name = f"{video_name_stem}_{timestamp}"
    
    frames_dir = Path(args.output) / unique_folder_name
    
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Created new output directory: {frames_dir}")

    print(f"[INFO] Extracting frames from {args.input}...")
    frame_paths_result = extract_video_frames(args.input, str(frames_dir), args.fps)

    frame_files = sorted([p for p in os.listdir(frames_dir) if p.lower().endswith(('.jpg', '.png'))])
    frame_paths = [str(frames_dir / f) for f in frame_files]

    print(f"[INFO] Extracted {len(frame_paths)} frames to {frames_dir}")

    print(f"[INFO] Running detection pipeline...")
    print(f"  - Person confidence: {args.person_conf}")
    print(f"  - Bat confidence: {args.bat_conf}")
    print(f"  - IoU threshold: {args.iou_thresh}")
    print(f"  - Consecutive frames required: {args.consec_frames}")
    print(f"  - Wicket confidence: {args.wicket_conf}")
    print(f"  - Pad confidence: {args.pad_conf}")
    print(f"  - Dynamic wicket detection: {'ON' if dynamic_wicket_detection else 'OFF'}")


    process_frames_pipeline(
        frame_paths,
        person_conf=args.person_conf,
        bat_conf=args.bat_conf,
        pad_conf=args.pad_conf,
        iou_thresh=args.iou_thresh,
        consec_required=args.consec_frames,
        wicket_conf=args.wicket_conf,
        preprocess=GLOBAL_CONFIG['enable_preprocessing'],
        display=GLOBAL_CONFIG['display_frames'],
        dynamic_wicket_detection=dynamic_wicket_detection,
        video_stem=video_name_stem,
    )

    print(f"[INFO] Detection pipeline completed.")

    print(f"[INFO] Creating output video...")
    output_video_path = frames_dir / f"{video_name_stem}_output.mp4"
    frames_to_video_with_custom_path(str(frames_dir), str(output_video_path), args.fps)
    print(f"[INFO] Output video saved to {output_video_path}")

if __name__ == "__main__":
    main()
