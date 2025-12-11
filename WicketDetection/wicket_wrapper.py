import cv2
import argparse
import sys
import os

from wicket_detector import detect_wicket

def main():
    # 1. Setup Argument Parser to get video path from terminal
    parser = argparse.ArgumentParser(description="Run Wicket Tracker on a video file.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    
    args = parser.parse_args()

    # 2. Verify video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: The file '{args.video_path}' does not exist.")
        sys.exit(1)

    # 3. Open the video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit(1)

    print(f"Processing video: {args.video_path} ...")
    print("Press 'q' to quit.")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video reached.")
            break
        
        frame_count += 1

        # 4. Pass frame to the tracker
        # The tracker modifies 'processed_frame' in-place (drawing boxes) 
        # and returns the list of detections.
        processed_frame, detections = detect_wicket(frame, conf=args.conf)

        # Optional: Print bbox details to terminal for debugging
        if detections:
            print(f"Frame {frame_count}: Found {len(detections)} objects")
            for det in detections:
                print(f"  - {det['label']}: Box {det['box']} | Conf: {det['conf']:.2f}")

        # 5. Display the frame with bounding boxes
        cv2.imshow("Wicket Tracker Output", processed_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()