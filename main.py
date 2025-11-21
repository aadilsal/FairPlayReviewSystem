import cv2
import os
import time

# Fix for OpenMP duplicate library warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# HSV values for cricket ball (customize as needed)
hsv_vals = {
    "hmin": 10,
    "smin": 44,
    "vmin": 192,
    "hmax": 125,
    "smax": 114,
    "vmax": 255,
}
from detection.person_detector import detect_persons
from detection.pose_detector import estimate_pose

def extract_video_frames(video_path, output_frames_folder, target_fps=30):
    """Convenience wrapper that delegates to the pipeline extractor.

    Keeps the original function name for backwards compatibility while using
    the refactored `pipeline.preprocessing.extract_video_frames` implementation.
    """
    from pipeline.preprocessing import extract_video_frames as _impl
    return _impl(video_path, output_frames_folder, target_fps)

def run_person_and_pose_detection_on_frames(frame_paths):
    """
    For each frame, run person detection and pose estimation.
    Uses marker files to avoid redundant processing.
    """
    print("[INFO] Running person detection and pose estimation...")
    for frame_path in frame_paths:
        pose_marker = frame_path + ".pose"
        if os.path.exists(pose_marker):
            print(f"[INFO] Pose already estimated for {frame_path}, skipping.")
            continue

        person_marker = frame_path + ".person"
        # --- CHECK: If person already detected, skip detection ---
        if not os.path.exists(person_marker):
            frame = cv2.imread(frame_path)
            frame_with_persons, detections = detect_persons(frame)
            cv2.imwrite(frame_path, frame_with_persons)
            with open(person_marker, "w") as f:
                f.write("person detected")
        else:
            frame_with_persons = cv2.imread(frame_path)

        frame_with_pose, keypoints = estimate_pose(frame_with_persons)
        print(f"[DEBUG] {frame_path} -> {len(keypoints)} persons detected with skeletons")
        #cv2.imshow("Pose Estimation", frame_with_pose)
        cv2.imwrite(frame_path, frame_with_pose)
        with open(pose_marker, "w") as f:
            f.write("pose estimated")
        cv2.waitKey(0)  # press any key for next frame

    cv2.destroyAllWindows()
    print("[INFO] Person detection and pose estimation completed.")

class CricketBallTracker:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Cricket ball detection/tracking functionality has been removed. "
            "Backups are available in `backup_ball_detection/` if you need to restore it."
        )


class CricketBallDetector:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Cricket ball detection has been removed from the codebase. "
            "Backups are available in `backup_ball_detection/` if you need to restore it."
        )
import argparse
from pipeline.preprocessing import extract_video_frames
from pipeline.main_pipeline import process_frames_pipeline
from pipeline.postprocessing import frames_to_video_with_custom_path


def process_single_video(input_path, output_dir, fps, motion_prediction=True, motion_preset='balanced',
                        enable_preprocessing=True, target_brightness=0.5):
    frame_paths, frames_dir = extract_video_frames(input_path, output_dir, fps)
    process_frames_pipeline(frame_paths, enable_motion_prediction=motion_prediction, motion_preset=motion_preset,
                          enable_preprocessing=enable_preprocessing, target_brightness=target_brightness)
    output_video_path = frames_to_video_with_custom_path(input_path, frames_dir, fps, output_dir)
    print(f"Output video saved to: {output_video_path}")


def process_folder(folder_path, output_dir, fps, motion_prediction=True, motion_preset='balanced',
                   enable_preprocessing=True, target_brightness=0.5):
    videos = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
              if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not videos:
        print(f"No video files found in {folder_path}")
        return
    for vid in videos:
        print(f"\n=== Processing {vid} ===")
        process_single_video(vid, output_dir, fps, motion_prediction, motion_preset,
                           enable_preprocessing, target_brightness)


def main():
    parser = argparse.ArgumentParser(description='Run FairPlayReviewSystem pipeline')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to input video file or folder containing videos')
    parser.add_argument('--output', '-o', default='outputs', help='Output directory')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS for extraction and output video')
    parser.add_argument('--no-motion-prediction', action='store_true', 
                        help='Disable ball motion prediction (default: enabled)')
    parser.add_argument('--motion-preset', default='balanced',
                        choices=['conservative', 'balanced', 'aggressive', 'high_quality', 'disabled'],
                        help='Motion prediction preset (default: balanced)')
    parser.add_argument('--no-preprocessing', action='store_true',
                        help='Disable adaptive frame preprocessing (default: enabled)')
    parser.add_argument('--target-brightness', type=float, default=0.5,
                        help='Target brightness for preprocessing (0.3-0.7, default: 0.5)')
    args = parser.parse_args()

    # Determine motion prediction settings
    enable_motion = not args.no_motion_prediction
    motion_preset = args.motion_preset if enable_motion else 'disabled'
    
    # Determine preprocessing settings
    enable_preprocessing = not args.no_preprocessing
    target_brightness = args.target_brightness
    
    print("\n" + "="*70)
    print("FAIRPLAY REVIEW SYSTEM - PIPELINE CONFIGURATION")
    print("="*70)
    print(f"Motion Prediction: {'✓ Enabled' if enable_motion else '✗ Disabled'}")
    if enable_motion:
        print(f"  └─ Preset: {motion_preset}")
    print(f"Frame Preprocessing: {'✓ Enabled' if enable_preprocessing else '✗ Disabled'}")
    if enable_preprocessing:
        print(f"  └─ Target Brightness: {target_brightness}")
    print("="*70 + "\n")

    if os.path.isdir(args.input):
        process_folder(args.input, args.output, args.fps, enable_motion, motion_preset,
                      enable_preprocessing, target_brightness)
    else:
        process_single_video(args.input, args.output, args.fps, enable_motion, motion_preset,
                           enable_preprocessing, target_brightness)


if __name__ == '__main__':
    main()