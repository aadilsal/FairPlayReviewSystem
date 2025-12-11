import argparse
import os
import shutil
import cv2
from pathlib import Path
from frame_extractor import extract_video_frames
from video_utils import frames_to_video_with_custom_path
from wicket_detector import detect_wicket

def main():
    parser = argparse.ArgumentParser(description='Run Standalone Wicket Detection')
    parser.add_argument('--input', '-i', required=True, help='Path to input video file')
    parser.add_argument('--output', '-o', default='outputs/wicket_test', help='Output directory')
    parser.add_argument('--fps', type=int, default=30, help='FPS for output video')
    parser.add_argument('--conf', type=float, default=0.25, help='Wicket detection confidence')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input video file not found: {args.input}")
        return

    video_name = Path(args.input).stem
    frames_dir = Path(args.output) / video_name
    
    # --- CRITICAL FIX: CLEANUP OLD FRAMES ---
    if os.path.exists(frames_dir):
        print(f"[INFO] Cleaning up old frames in {frames_dir}...")
        shutil.rmtree(frames_dir)
    
    frames_dir.mkdir(parents=True, exist_ok=True)
    # ----------------------------------------

    print(f"[INFO] Extracting FRESH frames from {args.input}...")
    extract_video_frames(args.input, str(frames_dir), args.fps)
    
    frame_files = sorted([p for p in os.listdir(frames_dir) if p.lower().endswith(('.jpg', '.png'))])
    print(f"[INFO] Processing {len(frame_files)} frames with Wicket Detector (conf={args.conf})...")

    for i, f_name in enumerate(frame_files):
        f_path = str(frames_dir / f_name)
        frame = cv2.imread(f_path)
        
        if frame is None:
            continue

        frame, detections = detect_wicket(frame, conf=args.conf)

        if i % 50 == 0:
            print(f"  Processed frame {i}/{len(frame_files)}")

        # Overwrite is okay now because we extracted FRESH frames at the start
        cv2.imwrite(f_path, frame)
        
        # Optional: Display (Press 'q' to quit early)
        cv2.imshow("Wicket Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    print(f"[INFO] Creating output video...")
    output_video_path = frames_dir / f"{video_name}_wicket.mp4"
    frames_to_video_with_custom_path(str(frames_dir), str(output_video_path), args.fps)
    print(f"[INFO] Done! Video saved at: {output_video_path}")

if __name__ == "__main__":
    main()