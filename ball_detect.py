import cv2
import numpy as np
import torch
from ultralytics import YOLO
import math
from typing import List, Tuple, Optional

class CricketBallDetector:
    def __init__(self, yolo_model_path: str = 'yolov8n.pt'):
        """
        Initialize the cricket ball detector
        
        Args:
            yolo_model_path: Path to YOLO model weights
        """
        self.yolo_model = YOLO(yolo_model_path)
        
        # Cricket ball color ranges in HSV
        # Red cricket ball color ranges
        self.red_lower1 = np.array([0, 120, 70])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 120, 70])
        self.red_upper2 = np.array([180, 255, 255])
        
        # White cricket ball color ranges
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 30, 255])
        
        # Sliding window parameters
        self.window_sizes = [(32, 32), (48, 48), (64, 64)]
        self.stride = 16
        
        # Ball detection confidence threshold
        self.confidence_threshold = 0.3
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better detection
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed image
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Enhance contrast
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def segment_by_color(self, image: np.ndarray) -> np.ndarray:
        """
        Segment potential ball regions using color information
        
        Args:
            image: Input BGR image
            
        Returns:
            Binary mask of potential ball regions
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create masks for red cricket ball
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Create mask for white cricket ball
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(red_mask, white_mask)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Dilate to ensure we capture the full ball region
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        combined_mask = cv2.dilate(combined_mask, kernel_dilate, iterations=1)
        
        return combined_mask
    
    def get_sliding_windows(self, image: np.ndarray, color_mask: Optional[np.ndarray] = None) -> List[Tuple]:
        """
        Generate sliding windows for ball detection
        
        Args:
            image: Input image
            color_mask: Optional color mask to limit search regions
            
        Returns:
            List of (x, y, width, height, window_image) tuples
        """
        h, w = image.shape[:2]
        windows = []
        
        for window_size in self.window_sizes:
            win_h, win_w = window_size
            
            for y in range(0, h - win_h + 1, self.stride):
                for x in range(0, w - win_w + 1, self.stride):
                    # If color mask is provided, check if window overlaps with mask
                    if color_mask is not None:
                        roi_mask = color_mask[y:y+win_h, x:x+win_w]
                        if np.sum(roi_mask) < (win_h * win_w * 0.1):  # Less than 10% overlap
                            continue
                    
                    window = image[y:y+win_h, x:x+win_w]
                    windows.append((x, y, win_w, win_h, window))
        
        return windows
    
    def classify_window_yolo(self, window: np.ndarray) -> float:
        """
        Classify a window using YOLO to detect if it contains a ball
        
        Args:
            window: Image window to classify
            
        Returns:
            Confidence score for ball detection
        """
        try:
            # Run YOLO detection on the window
            results = self.yolo_model(window, verbose=False)
            
            max_confidence = 0.0
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check for sports ball class (class 32 in COCO)
                        if int(box.cls) == 32:  # sports ball
                            confidence = float(box.conf)
                            max_confidence = max(max_confidence, confidence)
            
            return max_confidence
        except Exception as e:
            print(f"Error in YOLO classification: {e}")
            return 0.0
    
    def is_circular_object(self, contour: np.ndarray) -> Tuple[bool, float]:
        """
        Check if a contour represents a circular object (like a ball)
        
        Args:
            contour: OpenCV contour
            
        Returns:
            (is_circular, circularity_score)
        """
        if len(contour) < 5:
            return False, 0.0
        
        # Calculate contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return False, 0.0
        
        # Circularity = 4π * area / perimeter²
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        
        # Check if it's reasonably circular (ball-like)
        is_circular = circularity > 0.6 and area > 100
        
        return is_circular, circularity
    
    def detect_ball_sliding_window(self, image: np.ndarray, use_color_segmentation: bool = True) -> List[Tuple]:
        """
        Detect cricket ball using sliding window approach
        
        Args:
            image: Input BGR image
            use_color_segmentation: Whether to use color segmentation for optimization
            
        Returns:
            List of detected balls as (x, y, width, height, confidence) tuples
        """
        preprocessed = self.preprocess_image(image)
        color_mask = None
        
        if use_color_segmentation:
            color_mask = self.segment_by_color(preprocessed)
        
        # Get sliding windows
        windows = self.get_sliding_windows(preprocessed, color_mask)
        
        detections = []
        
        print(f"Processing {len(windows)} windows...")
        
        for i, (x, y, w, h, window) in enumerate(windows):
            if i % 100 == 0:
                print(f"Processed {i}/{len(windows)} windows")
            
            # Classify window using YOLO
            confidence = self.classify_window_yolo(window)
            
            if confidence > self.confidence_threshold:
                detections.append((x, y, w, h, confidence))
        
        # Apply Non-Maximum Suppression to remove overlapping detections
        detections = self.apply_nms(detections)
        
        return detections
    
    def apply_nms(self, detections: List[Tuple], iou_threshold: float = 0.3) -> List[Tuple]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        
        Args:
            detections: List of (x, y, w, h, confidence) tuples
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []
        
        # Convert to format expected by cv2.dnn.NMSBoxes
        boxes = [[x, y, w, h] for x, y, w, h, _ in detections]
        scores = [conf for _, _, _, _, conf in detections]
        
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, iou_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        else:
            return []
    
    def detect_ball_yolo_direct(self, image: np.ndarray) -> List[Tuple]:
        """
        Detect cricket ball using direct YOLO detection
        
        Args:
            image: Input BGR image
            
        Returns:
            List of detected balls as (x, y, width, height, confidence) tuples
        """
        results = self.yolo_model(image, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls) == 32:  # sports ball class
                        confidence = float(box.conf)
                        if confidence > self.confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                            detections.append((x, y, w, h, confidence))
        
        return detections
    
    def detect_ball(self, image: np.ndarray, method: str = 'region_based') -> List[Tuple]:
        """
        Main function to detect cricket ball with optimized methods
        
        Args:
            image: Input BGR image
            method: Detection method ('region_based', 'sliding_window', 'color_optimized', 'yolo_direct')
            
        Returns:
            List of detected balls as (x, y, width, height, confidence) tuples
        """
        if method == 'region_based':
            return self.detect_ball_region_based(image)
        elif method == 'sliding_window':
            return self.detect_ball_sliding_window(image, use_color_segmentation=False)
        elif method == 'color_optimized':
            return self.detect_ball_sliding_window(image, use_color_segmentation=True)
        elif method == 'yolo_direct':
            return self.detect_ball_yolo_direct(image)
        else:
            raise ValueError(f"Unknown method: {method}. Available: 'region_based', 'sliding_window', 'color_optimized', 'yolo_direct'")
    
    def visualize_detections(self, image: np.ndarray, detections: List[Tuple], 
                            frame_number: int = -1, show_tracking_info: bool = True) -> np.ndarray:
        """
        Draw enhanced visualizations around detected balls
        
        Args:
            image: Input image
            detections: List of detections
            frame_number: Frame number for display (-1 if not applicable)
            show_tracking_info: Whether to show tracking information
            
        Returns:
            Image with enhanced ball detection visualizations
        """
        result_image = image.copy()
        h, w = result_image.shape[:2]
        
        # Add semi-transparent overlay for better text visibility
        overlay = result_image.copy()
        
        for i, (x, y, w_box, h_box, confidence) in enumerate(detections):
            # Calculate center point
            center_x = x + w_box // 2
            center_y = y + h_box // 2
            
            # Choose colors based on confidence
            if confidence > 0.7:
                box_color = (0, 255, 0)    # Green for high confidence
                circle_color = (0, 255, 0)
                text_color = (0, 255, 0)
            elif confidence > 0.4:
                box_color = (0, 255, 255)  # Yellow for medium confidence
                circle_color = (0, 255, 255)
                text_color = (0, 255, 255)
            else:
                box_color = (0, 165, 255)  # Orange for low confidence
                circle_color = (0, 165, 255)
                text_color = (0, 165, 255)
            
            # Draw bounding box with thicker lines
            cv2.rectangle(result_image, (x, y), (x + w_box, y + h_box), box_color, 3)
            
            # Draw center point
            cv2.circle(result_image, (center_x, center_y), 8, circle_color, -1)
            cv2.circle(result_image, (center_x, center_y), 12, circle_color, 2)
            
            # Draw crosshair at center
            cv2.line(result_image, (center_x - 15, center_y), (center_x + 15, center_y), circle_color, 2)
            cv2.line(result_image, (center_x, center_y - 15), (center_x, center_y + 15), circle_color, 2)
            
            # Add ball number and confidence
            label = f"BALL #{i+1}"
            confidence_text = f"Conf: {confidence:.3f}"
            position_text = f"Pos: ({center_x},{center_y})"
            
            # Create text background for better visibility
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            conf_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            pos_size = cv2.getTextSize(position_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            
            # Position text above the bounding box
            text_y_start = max(y - 60, 20)
            
            # Draw text backgrounds
            cv2.rectangle(overlay, (x, text_y_start), (x + max(label_size[0], conf_size[0], pos_size[0]) + 10, text_y_start + 50), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(result_image, label, (x + 5, text_y_start + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            cv2.putText(result_image, confidence_text, (x + 5, text_y_start + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.putText(result_image, position_text, (x + 5, text_y_start + 47), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        # Blend overlay for semi-transparent text backgrounds
        result_image = cv2.addWeighted(result_image, 0.8, overlay, 0.2, 0)
        
        if show_tracking_info:
            # Add frame information and detection summary
            info_text = []
            if frame_number >= 0:
                info_text.append(f"Frame: {frame_number}")
            info_text.append(f"Balls Detected: {len(detections)}")
            
            if detections:
                avg_conf = sum(det[4] for det in detections) / len(detections)
                info_text.append(f"Avg Confidence: {avg_conf:.3f}")
            
            # Draw info panel
            panel_height = len(info_text) * 25 + 20
            cv2.rectangle(result_image, (10, 10), (280, panel_height), (0, 0, 0), -1)
            cv2.rectangle(result_image, (10, 10), (280, panel_height), (255, 255, 255), 2)
            
            for i, text in enumerate(info_text):
                cv2.putText(result_image, text, (20, 35 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add detection status indicator
            if detections:
                cv2.circle(result_image, (250, 25), 8, (0, 255, 0), -1)  # Green dot
                cv2.putText(result_image, "ACTIVE", (190, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                cv2.circle(result_image, (250, 25), 8, (0, 0, 255), -1)  # Red dot
                cv2.putText(result_image, "NO BALL", (185, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return result_image
