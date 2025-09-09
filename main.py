import cv2
import os
import argparse
import time
from pathlib import Path
from ball_detect import CricketBallDetector
from frame_extractor import extract_frames

class CricketBallTracker:
    def __init__(self, yolo_model_path='yolov8n.pt'):
        """
        Initialize the cricket ball tracking system
        
        Args:
            yolo_model_path: Path to YOLO model weights
        """
        self.detector = CricketBallDetector(yolo_model_path)
        self.results = []
    
    def process_single_image(self, image_path, output_dir="outputs/images", methods=['color_optimized']):
        """
        Process a single image for ball detection
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            methods: List of detection methods to use
        """
        print(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process with each method
        for method in methods:
            print(f"  Using {method} method...")
            start_time = time.time()
            
            # Detect balls
            detections = self.detector.detect_ball(image, method=method)
            processing_time = time.time() - start_time
            
            print(f"  Found {len(detections)} ball(s) in {processing_time:.2f}s")
            
            # Visualize results
            result_image = self.detector.visualize_detections(image, detections)
            
            # Save result
            image_name = Path(image_path).stem
            output_path = os.path.join(output_dir, f"{image_name}_{method}.jpg")
            cv2.imwrite(output_path, result_image)
            print(f"  Result saved to {output_path}")
            
            # Store results
            self.results.append({
                'image_path': image_path,
                'method': method,
                'detections': detections,
                'processing_time': processing_time,
                'output_path': output_path
            })
    
    def process_video(self, video_path, output_dir="outputs/video", 
                     frame_output_dir="outputs/frames", target_fps=10, 
                     methods=['color_optimized']):
        """
        Process a video for ball detection
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save video results
            frame_output_dir: Directory to save extracted frames
            target_fps: FPS for frame extraction
            methods: List of detection methods to use
        """
        print(f"Processing video: {video_path}")
        
        # Extract frames from video
        print("Extracting frames...")
        frame_paths = extract_frames(video_path, frame_output_dir, target_fps)
        print(f"Extracted {len(frame_paths)} frames")
        
        if not frame_paths:
            print("Error: No frames extracted from video")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process frames for each method
        for method in methods:
            print(f"\nProcessing frames with {method} method...")
            method_output_dir = os.path.join(output_dir, method)
            os.makedirs(method_output_dir, exist_ok=True)
            
            processed_frames = []
            total_detections = 0
            total_time = 0
            
            for i, frame_path in enumerate(frame_paths):
                if i % 10 == 0:  # Progress update every 10 frames
                    print(f"  Processing frame {i+1}/{len(frame_paths)}")
                
                # Load frame
                frame = cv2.imread(frame_path)
                if frame is None:
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
        """
        Create a video from processed frames
        
        Args:
            frame_paths: List of processed frame paths
            output_video_path: Output video file path
            fps: Frames per second for output video
        """
        if not frame_paths:
            return
        
        print(f"Creating output video: {output_video_path}")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            print("Error: Could not read first frame for video creation")
            return
        
        height, width, layers = first_frame.shape
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Write frames to video
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)
        
        out.release()
        print(f"Output video saved to: {output_video_path}")
    
    def generate_report(self, output_file="detection_report.txt"):
        """
        Generate a summary report of all detections
        
        Args:
            output_file: Path to save the report
        """
        if not self.results:
            print("No results to report")
            return
        
        print(f"Generating report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("Cricket Ball Detection Report\n")
            f.write("=" * 40 + "\n\n")
            
            # Group results by method
            methods = {}
            for result in self.results:
                method = result['method']
                if method not in methods:
                    methods[method] = []
                methods[method].append(result)
            
            # Write summary for each method
            for method, results in methods.items():
                f.write(f"Method: {method.upper()}\n")
                f.write("-" * 20 + "\n")
                
                total_detections = sum(len(r['detections']) for r in results)
                avg_time = sum(r['processing_time'] for r in results) / len(results)
                
                f.write(f"Total frames/images processed: {len(results)}\n")
                f.write(f"Total ball detections: {total_detections}\n")
                f.write(f"Average processing time: {avg_time:.3f}s\n")
                f.write(f"Detection rate: {total_detections/len(results):.2f} balls per frame\n\n")
                
                # List frames with detections
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
                       help='Path to input image or video file')
    parser.add_argument('--output', '-o', default='outputs',
                       help='Output directory (default: outputs)')
    parser.add_argument('--methods', '-m', nargs='+', 
                       choices=['sliding_window', 'color_optimized', 'yolo_direct'],
                       default=['color_optimized'],
                       help='Detection methods to use')
    parser.add_argument('--fps', type=int, default=10,
                       help='Target FPS for video frame extraction (default: 10)')
    parser.add_argument('--model', default='yolov8n.pt',
                       help='Path to YOLO model weights (default: yolov8n.pt)')
    parser.add_argument('--report', action='store_true',
                       help='Generate detection report')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = CricketBallTracker(args.model)
    
    # Check if input is video or image
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {args.input} does not exist")
        return
    
    # Determine file type
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    file_extension = input_path.suffix.lower()
    
    if file_extension in video_extensions:
        # Process video
        print("Detected video input")
        tracker.process_video(
            video_path=args.input,
            output_dir=os.path.join(args.output, 'video'),
            frame_output_dir=os.path.join(args.output, 'frames'),
            target_fps=args.fps,
            methods=args.methods
        )
    elif file_extension in image_extensions:
        # Process single image
        print("Detected image input")
        tracker.process_single_image(
            image_path=args.input,
            output_dir=os.path.join(args.output, 'images'),
            methods=args.methods
        )
    else:
        print(f"Error: Unsupported file format {file_extension}")
        print(f"Supported video formats: {', '.join(video_extensions)}")
        print(f"Supported image formats: {', '.join(image_extensions)}")
        return
    
    # Generate report if requested
    if args.report:
        report_path = os.path.join(args.output, 'detection_report.txt')
        tracker.generate_report(report_path)
    
    print("\nProcessing complete!")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()