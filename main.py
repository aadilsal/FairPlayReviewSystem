import cv2
import os
import argparse
import time
from pathlib import Path
from ball_detect import CricketBallDetector
from frame_extractor import extract_frames
from ball_tracker import ball_detect
from yolo_detect import YOLOBallDetector

# HSV values for cricket ball (customize as needed)
hsv_vals = {
    "hmin": 10,
    "smin": 44,
    "vmin": 192,
    "hmax": 125,
    "smax": 114,
    "vmax": 255,
}
from person_detector import detect_persons
from pose_estimator import estimate_pose

def extract_video_frames(video_path, output_frames_folder, target_fps=30):
    """
    Extract frames from a video at the specified FPS.
    Only extracts if frames don't already exist.
    Returns a list of frame paths.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_frames_folder, video_name)
    # --- CHECK: If frames already extracted, skip extraction ---
    if os.path.exists(video_output_folder):
        existing_frames = [f for f in os.listdir(video_output_folder) if f.lower().endswith('.jpg')]
        if existing_frames:
            print(f"Frames already exist for {video_name}, skipping extraction.")
            return [os.path.join(video_output_folder, f) for f in sorted(existing_frames)]
    os.makedirs(video_output_folder, exist_ok=True)
    frame_paths = extract_frames(video_path, output_folder=video_output_folder, target_fps=target_fps)
    print(f"Extracted {len(frame_paths)} frames from {video_name} to {video_output_folder}")
    return frame_paths

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
    def __init__(self, yolo_model_path='yolov8n.pt'):
        """
        Initialize the cricket ball tracking system
        
        Args:
            yolo_model_path: Path to YOLO model weights
        """
        self.detector = CricketBallDetector(yolo_model_path)
        self.results = []

    def process_video(self, video_path, output_dir="outputs/video", 
                     frame_output_dir="outputs/frames", target_fps=10, 
                     methods=['color_optimized']):
        print(f"Processing video: {video_path}")
        print("Extracting frames...")
        frame_paths = extract_frames(video_path, frame_output_dir, target_fps)
        print(f"Extracted {len(frame_paths)} frames")
        if not frame_paths:
            print("Error: No frames extracted from video")
            return
        os.makedirs(output_dir, exist_ok=True)
        for method in methods:
            print(f"\nProcessing frames with {method} method...")
            method_output_dir = os.path.join(output_dir, method)
            os.makedirs(method_output_dir, exist_ok=True)
            processed_frames = []
            total_detections = 0
            total_time = 0
            for i, frame_path in enumerate(frame_paths):
                if i % 10 == 0:
                    print(f"  Processing frame {i+1}/{len(frame_paths)}")
                frame = cv2.imread(frame_path)
                if frame is None:
                    print(f"Error: Could not load frame {frame_path}")
                    continue
                
                start_time = time.time()
                
                # Detect balls
                detections = self.detector.detect_ball(frame, method=method)
                processing_time = time.time() - start_time
                
                total_detections += len(detections)
                total_time += processing_time
                
                # Visualize results
                result_frame = self.detector.visualize_detections(frame, detections)
                
                # Save processed frame
                frame_name = f"processed_frame_{i:04d}.jpg"
                output_frame_path = os.path.join(method_output_dir, frame_name)
                cv2.imwrite(output_frame_path, result_frame)
                processed_frames.append(output_frame_path)
                
                # Store frame results
                self.results.append({
                    'frame_path': frame_path,
                    'frame_number': i,
                    'method': method,
                    'detections': detections,
                    'processing_time': processing_time,
                    'output_path': output_frame_path
                })
            
            print(f"  Method {method} complete:")
            print(f"    Total detections: {total_detections}")
            print(f"    Average processing time: {total_time/len(frame_paths):.3f}s per frame")
            print(f"    Processed frames saved to: {method_output_dir}")
            
            # Create output video from processed frames
            self.create_output_video(processed_frames, 
                                   os.path.join(output_dir, f"output_{method}.mp4"),
                                   target_fps)

    def create_output_video(self, frame_paths, output_video_path, fps=10):
        if not frame_paths:
            return
        print(f"Creating output video: {output_video_path}")
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            print("Error: Could not read first frame for video creation")
            return
        height, width, layers = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)
        out.release()
        print(f"Output video saved to: {output_video_path}")

    def generate_report(self, output_file="detection_report.txt"):
        if not self.results:
            print("No results to report")
            return
        print(f"Generating report: {output_file}")
        with open(output_file, 'w') as f:
            f.write("Cricket Ball Detection Report\n")
            f.write("=" * 40 + "\n\n")
            methods = {}
            for result in self.results:
                method = result['method']
                if method not in methods:
                    methods[method] = []
                methods[method].append(result)
            for method, results in methods.items():
                f.write(f"Method: {method.upper()}\n")
                f.write("-" * 20 + "\n")
                total_detections = sum(len(r['detections']) for r in results)
                avg_time = sum(r['processing_time'] for r in results) / len(results)
                f.write(f"Total frames/images processed: {len(results)}\n")
                f.write(f"Total ball detections: {total_detections}\n")
                f.write(f"Average processing time: {avg_time:.3f}s\n")
                f.write(f"Detection rate: {total_detections/len(results):.2f} balls per frame\n\n")
                frames_with_balls = [r for r in results if len(r['detections']) > 0]
                if frames_with_balls:
                    f.write("Frames with ball detections:\n")
                    for result in frames_with_balls:
                        frame_info = result.get('frame_number', 'N/A')
                        f.write(f"  Frame {frame_info}: {len(result['detections'])} ball(s)\n")
                f.write("\n")
        print(f"Report saved to: {output_file}")
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