import cv2
import os
import json
from detection.person_detector import detect_persons
from detection.pose_detector import estimate_pose
from detection.detect_balls import detect_ball_on_image, detect_ball_on_image_adaptive, draw_boxes_on_image, detect_balls
from detection.motion_config import get_config

# Default ball detector model and annotation output
BALL_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yolov8n.pt')
ANNOTATED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs', 'annotated')

# Motion prediction configuration (can be changed to 'conservative', 'aggressive', etc.)
MOTION_PREDICTION_PRESET = 'balanced'

# Frame preprocessing configuration (NEW)
ENABLE_PREPROCESSING = True  # Toggle frame preprocessing on/off
TARGET_BRIGHTNESS = 0.5      # Target brightness for normalization (0.3-0.7)

# Cache for batch ball detections
_BALL_DETECTIONS_CACHE = None

# Statistics tracking for preprocessing
_PREPROCESSING_STATS = {
    'total_frames': 0,
    'preprocessed_frames': 0,
    'quality_scores': [],
    'preprocessing_times': []
}


def _ensure_dir(d):
    if d:
        os.makedirs(d, exist_ok=True)

def process_frames_pipeline(frame_paths, enable_motion_prediction=True, motion_preset='balanced', 
                          enable_preprocessing=True, target_brightness=0.5):
    """Process frames with person, pose, and ball detection.
    
    Args:
        frame_paths: List of frame file paths
        enable_motion_prediction: Enable ball motion prediction to fill gaps
        motion_preset: Preset for motion prediction ('conservative', 'balanced', 'aggressive', 'high_quality', 'disabled')
        enable_preprocessing: Enable adaptive frame preprocessing for robust detection (NEW)
        target_brightness: Target brightness for preprocessing normalization (NEW)
    """
    global _BALL_DETECTIONS_CACHE, _PREPROCESSING_STATS
    
    # Reset preprocessing statistics
    _PREPROCESSING_STATS = {
        'total_frames': 0,
        'preprocessed_frames': 0,
        'quality_scores': [],
        'preprocessing_times': []
    }
    
    # Log preprocessing configuration
    if enable_preprocessing:
        print(f"[INFO] 🎨 Frame Preprocessing: ENABLED")
        print(f"[INFO]    Target Brightness: {target_brightness}")
        print(f"[INFO]    Quality Threshold: 0.6 (frames below this will be enhanced)")
    else:
        print(f"[INFO] Frame Preprocessing: DISABLED")
    
    # First, run batch ball detection with motion prediction on all frames
    if enable_motion_prediction and motion_preset != 'disabled':
        print(f"[INFO] Running batch ball detection with motion prediction (preset: {motion_preset})...")
        frames_dir = os.path.dirname(frame_paths[0]) if frame_paths else ''
        
        if frames_dir and os.path.exists(frames_dir):
            try:
                # Get motion prediction config
                motion_config = get_config(motion_preset)
                
                # Run batch detection with motion prediction
                ball_records = detect_balls(
                    frames_dir=frames_dir,
                    model_path=BALL_MODEL,
                    output_csv=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs', 'ball_detections.csv'),
                    output_json=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs', 'ball_detections.json'),
                    annotated_dir=ANNOTATED_DIR,
                    batch_size=16,
                    device='cpu',
                    target_class_names=['sports ball'],
                    verbose=True,
                    **motion_config  # Apply motion prediction configuration
                )
                
                # Create frame index to detection mapping
                frame_to_detections = {}
                for record in ball_records:
                    frame_id = record.get('frame_id', '')
                    if frame_id:
                        if frame_id not in frame_to_detections:
                            frame_to_detections[frame_id] = []
                        frame_to_detections[frame_id].append(record)
                
                # Store for later use in frame processing
                _BALL_DETECTIONS_CACHE = frame_to_detections
                print(f"[INFO] Ball detection complete: {len(ball_records)} total detections (including predictions)")
            except Exception as e:
                print(f"[WARN] Batch ball detection with motion prediction failed: {e}")
                print(f"[INFO] Falling back to per-frame detection...")
                _BALL_DETECTIONS_CACHE = None
        else:
            _BALL_DETECTIONS_CACHE = None
    else:
        print(f"[INFO] Motion prediction disabled, using per-frame ball detection...")
        _BALL_DETECTIONS_CACHE = None
    
    # Process each frame for person and pose detection
    for frame_path in frame_paths:
        pose_marker = frame_path + ".pose"
        if os.path.exists(pose_marker):
            print(f"[INFO] All detections already done for {frame_path}, skipping.")
            continue

        person_marker = frame_path + ".person"
        ball_marker = frame_path + ".ball"

        frame = cv2.imread(frame_path)

        # Ball detection: use batch results if available, otherwise fallback to per-frame
        frame_with_ball = frame
        frame_basename = os.path.basename(frame_path)
        
        # Check if we have batch detection results
        if _BALL_DETECTIONS_CACHE is not None:
            # Use pre-computed batch detections (with motion prediction)
            detections = _BALL_DETECTIONS_CACHE.get(frame_basename, [])
            
            if detections:
                # Annotated image should already exist from batch processing
                annotated_path = os.path.join(ANNOTATED_DIR, frame_basename)
                if os.path.exists(annotated_path):
                    frame_with_ball = cv2.imread(annotated_path)
                else:
                    # Fallback: draw boxes manually
                    frame_with_ball = frame.copy()
                    names = [d['class_name'] for d in detections]
                    confs = [d['confidence'] for d in detections]
                    boxes = [(d['x_min'], d['y_min'], d['x_max'], d['y_max']) for d in detections]
                    types = [d.get('detection_type', 'detected') for d in detections]
                    draw_boxes_on_image(frame_with_ball, boxes, confs, names, detection_types=types)
                
                with open(ball_marker, 'w') as f:
                    f.write(json.dumps(detections))
            else:
                with open(ball_marker, 'w') as f:
                    f.write('no_ball')
        else:
            # Fallback: per-frame detection with adaptive preprocessing (NEW)
            try:
                if enable_preprocessing:
                    # Use adaptive detection with preprocessing
                    result = detect_ball_on_image_adaptive(
                        image=frame,
                        model_path=BALL_MODEL,
                        device='cpu',
                        target_class_names=['sports ball'],
                        enable_preprocessing=True,
                        target_brightness=target_brightness,
                        log_enhancements=False,  # Avoid per-frame logging
                        verbose=False
                    )
                    
                    detections = result['detections']
                    
                    # Track preprocessing statistics
                    _PREPROCESSING_STATS['total_frames'] += 1
                    _PREPROCESSING_STATS['quality_scores'].append(result['quality_score'])
                    if result['was_preprocessed']:
                        _PREPROCESSING_STATS['preprocessed_frames'] += 1
                        _PREPROCESSING_STATS['preprocessing_times'].append(result['preprocessing_time_ms'])
                    
                    # Log frame processing details
                    quality_status = "✓" if result['quality_score'] >= 0.6 else "⚠"
                    preprocess_status = "🎨" if result['was_preprocessed'] else "  "
                    print(f"[INFO] {preprocess_status} {quality_status} {frame_basename}: "
                          f"Quality={result['quality_score']:.2f}, "
                          f"Detections={len(detections)}, "
                          f"Conf={result['confidence_threshold']:.2f}, "
                          f"Time={result['total_time_ms']:.1f}ms")
                    
                    if result['was_preprocessed'] and result['enhancements_applied']:
                        enhancements_str = ', '.join(result['enhancements_applied'])
                        print(f"[INFO]    └─ Enhanced: {enhancements_str} "
                              f"({result['preprocessing_time_ms']:.1f}ms)")
                else:
                    # Standard detection without preprocessing
                    detections = detect_ball_on_image(frame, model_path=BALL_MODEL, device='cpu', verbose=False)
                
                if detections:
                    frame_with_ball = frame.copy()
                    names = [d['class_name'] for d in detections]
                    confs = [d['confidence'] for d in detections]
                    boxes = [(d['x_min'], d['y_min'], d['x_max'], d['y_max']) for d in detections]
                    draw_boxes_on_image(frame_with_ball, boxes, confs, names)
                    _ensure_dir(ANNOTATED_DIR)
                    out_name = os.path.join(ANNOTATED_DIR, frame_basename)
                    cv2.imwrite(out_name, frame_with_ball)
                    with open(ball_marker, 'w') as f:
                        f.write(json.dumps(detections))
                else:
                    with open(ball_marker, 'w') as f:
                        f.write('no_ball')
            except Exception as e:
                print(f"[WARN] Ball detection failed for {frame_path}: {e}")
                frame_with_ball = frame

        # Person detection
        if not os.path.exists(person_marker):
            frame_with_persons, _ = detect_persons(frame_with_ball)
            cv2.imwrite(frame_path, frame_with_persons)
            with open(person_marker, "w") as f:
                f.write("person detected")
        else:
            frame_with_persons = cv2.imread(frame_path)

        # Pose estimation
        frame_with_pose, _ = estimate_pose(frame_with_persons)
        cv2.imwrite(frame_path, frame_with_pose)
        with open(pose_marker, "w") as f:
            f.write("pose estimated")

        # --- Display the processed frame ---
        cv2.imshow("Processed Frame", frame_with_pose)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
    # Print preprocessing statistics summary
    if enable_preprocessing and _PREPROCESSING_STATS['total_frames'] > 0:
        print("\n" + "="*70)
        print("📊 PREPROCESSING STATISTICS SUMMARY")
        print("="*70)
        
        total = _PREPROCESSING_STATS['total_frames']
        preprocessed = _PREPROCESSING_STATS['preprocessed_frames']
        quality_scores = _PREPROCESSING_STATS['quality_scores']
        preprocess_times = _PREPROCESSING_STATS['preprocessing_times']
        
        preprocessing_rate = (preprocessed / total * 100) if total > 0 else 0
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        avg_preprocess_time = sum(preprocess_times) / len(preprocess_times) if preprocess_times else 0
        
        print(f"Total Frames Processed: {total}")
        print(f"Frames Preprocessed: {preprocessed} ({preprocessing_rate:.1f}%)")
        print(f"Average Quality Score: {avg_quality:.2f}")
        
        if preprocess_times:
            print(f"Average Preprocessing Time: {avg_preprocess_time:.1f}ms")
            print(f"Total Preprocessing Time: {sum(preprocess_times):.0f}ms")
        
        # Quality distribution
        if quality_scores:
            poor = sum(1 for q in quality_scores if q < 0.4)
            fair = sum(1 for q in quality_scores if 0.4 <= q < 0.6)
            good = sum(1 for q in quality_scores if 0.6 <= q < 0.8)
            excellent = sum(1 for q in quality_scores if q >= 0.8)
            
            print(f"\nQuality Distribution:")
            print(f"  Poor (<0.4):      {poor:3d} ({poor/total*100:5.1f}%)")
            print(f"  Fair (0.4-0.6):   {fair:3d} ({fair/total*100:5.1f}%)")
            print(f"  Good (0.6-0.8):   {good:3d} ({good/total*100:5.1f}%)")
            print(f"  Excellent (>0.8): {excellent:3d} ({excellent/total*100:5.1f}%)")
        
        print("="*70 + "\n")
    
    # Clean up batch detection cache
    _BALL_DETECTIONS_CACHE = None

