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