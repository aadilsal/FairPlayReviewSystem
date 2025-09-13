import cv2
import os
from cvzone.ColorModule import ColorFinder
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
                    continue
                start_time = time.time()
                detections = self.detector.detect_ball(frame, method=method)
                processing_time = time.time() - start_time
                total_detections += len(detections)
                total_time += processing_time
                result_frame = self.detector.visualize_detections(frame, detections)
                frame_name = f"processed_frame_{i:04d}.jpg"
                output_frame_path = os.path.join(method_output_dir, frame_name)
                cv2.imwrite(output_frame_path, result_frame)
                processed_frames.append(output_frame_path)
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

def main():
    parser = argparse.ArgumentParser(description='Cricket Ball Detection System')
    parser.add_argument('--input', '-i', required=True, 
                       help='Path to input video file')
    parser.add_argument('--output', '-o', default='outputs',
                       help='Output directory (default: outputs)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS for video frame extraction (default: 30)')
    parser.add_argument('--max_frames', type=int, default=3,
                       help='Number of frames to run pose estimation on (default: 3)')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {args.input} does not exist")
        return

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    file_extension = input_path.suffix.lower()

    if file_extension in video_extensions:
        # 1. Extract frames (if not already done)
        frame_paths = extract_video_frames(
            str(input_path),
            os.path.join(args.output, 'frames'),
            target_fps=args.fps
        )
        # 2. Run person and pose detection (if not already done)
        run_person_and_pose_detection_on_frames(
            frame_paths,
            #max_frames=args.max_frames
        )
        print("\nProcessing complete!")
        print(f"Frames and results saved to: {os.path.join(args.output, 'frames')}")
    else:
        print(f"Error: Only video files are supported for this workflow.")
        print(f"Supported video formats: {', '.join(video_extensions)}")
        return

if __name__ == "__main__":
    video_path = "test_videos\\lbw.mp4"  # Change to your video path
    frame_output_dir = "outputs/frames"
    os.makedirs(frame_output_dir, exist_ok=True)
    frame_paths = extract_frames(video_path, frame_output_dir, target_fps=10)
    print(f"Extracted {len(frame_paths)} frames")
    color_finder = ColorFinder(False)
    yolo_detector = YOLOBallDetector('outputs/yolov8_cricket_ball2/weights/best.pt')  # Use trained cricket ball model
    for frame_path in frame_paths:
        img = cv2.imread(frame_path)
        # First, try YOLO detection
        yolo_detections = yolo_detector.detect(img)
        if yolo_detections:
            # Draw YOLO detections
            for (x, y, w, h, confidence) in yolo_detections:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"YOLO {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Ball Detection", img)
        else:
            # Fallback to color-based detection
            img_contours, x, y = ball_detect(img, color_finder, hsv_vals)
            if img_contours is not None:
                cv2.imshow("Ball Detection", img_contours)
            else:
                cv2.imshow("Ball Detection", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()