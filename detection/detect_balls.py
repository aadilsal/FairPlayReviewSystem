"""Batch YOLO ball detector utility.

Usage:
    - Place this file in your project (e.g., `detection/detect_balls.py`).
    - Install `ultralytics` and `opencv-python` (and optionally `tqdm`).

This script provides functions to:
    - Load frame file paths from a directory
    - Run YOLO (ultralytics) inference in batches
    - Filter detections by target class name(s) or id(s)
    - Save detections to CSV and JSON
    - Optionally draw annotated images to a directory

The code is defensive against missing/corrupted images and reports progress.
"""
from typing import List, Optional, Sequence, Dict, Any
import os
import math
import json
import csv
import time
import traceback

try:
    import cv2
except Exception:
    raise ImportError("opencv-python is required. Install with: pip install opencv-python")
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    raise ImportError("ultralytics is required. Install with: pip install ultralytics")

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


def get_image_paths(frames_dir: str, exts: Sequence[str] = ('.jpg', '.jpeg', '.png')) -> List[str]:
    """Return sorted list of image file paths under `frames_dir`.

    Args:
        frames_dir: directory containing pre-extracted frames
        exts: allowed image extensions

    Returns:
        sorted list of file paths
    """
    frames = []
    for root, _, files in os.walk(frames_dir):
        for f in files:
            if f.lower().endswith(exts):
                frames.append(os.path.join(root, f))
    frames.sort()
    return frames


def ensure_dir(d: Optional[str]):
    if not d:
        return
    os.makedirs(d, exist_ok=True)


def write_outputs(records: List[Dict[str, Any]], csv_path: Optional[str], json_path: Optional[str]):
    """Save records list to CSV and/or JSON."""
    if csv_path:
        fieldnames = [
            'frame_index', 'frame_id', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class_id', 'class_name', 'detection_type'
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow({k: r.get(k, '') for k in fieldnames})

    if json_path:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)


def draw_boxes_on_image(img, boxes, confidences, class_names, color=(0, 255, 0), detection_types=None):
    """Draw bounding boxes and confidences on `img` (OpenCV BGR image).

    boxes: iterable of (x_min, y_min, x_max, y_max)
    confidences: iterable of float
    class_names: iterable of strings
    color: default color (BGR tuple) for detected boxes
    detection_types: optional iterable of 'detected'/'predicted' strings
                     If provided, predicted boxes are drawn in yellow/green
    """
    # Color scheme
    detected_color = color  # Green by default (0, 255, 0)
    predicted_color = (0, 255, 255)  # Yellow in BGR
    
    detection_types = detection_types or ['detected'] * len(boxes)
    
    for (x1, y1, x2, y2), conf, cname, det_type in zip(boxes, confidences, class_names, detection_types):
        x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
        
        # Choose color based on detection type
        box_color = predicted_color if det_type == 'predicted' else detected_color
        
        cv2.rectangle(img, (x1i, y1i), (x2i, y2i), box_color, 2)
        label = f"{cname} {conf:.2f}"
        if det_type == 'predicted':
            label += " (pred)"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (x1i, y1i - t_size[1] - 4), (x1i + t_size[0], y1i), box_color, -1)
        cv2.putText(img, label, (x1i, y1i - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def _safe_read(path: str, cache: Optional[Dict[str, Any]] = None):
    """Return image from cache or disk; None on failure."""
    if cache is not None:
        img = cache.get(path)
        if img is not None:
            return img
    try:
        img = cv2.imread(path)
        return img
    except Exception:
        return None


def _safe_imwrite(path: str, img) -> bool:
    """Write image to `path`, ensuring parent directory exists. Returns True on success."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        ok = cv2.imwrite(path, img)
        if not ok:
            print(f"Warning: cv2.imwrite failed for {path}")
        return bool(ok)
    except Exception as e:
        print(f"Warning: failed to write annotated image for {path}\n  {e}")
        return False


def detect_balls(
    frames_dir: str,
    model_path: str,
    output_csv: Optional[str] = 'detections.csv',
    output_json: Optional[str] = 'detections.json',
    annotated_dir: Optional[str] = None,
    batch_size: int = 16,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = 'cpu',
    target_class_names: Optional[Sequence[str]] = None,
    target_class_ids: Optional[Sequence[int]] = None,
    verbose: bool = True,
    enable_motion_prediction: bool = True,
    max_gap_frames: int = 5,
    prediction_confidence: float = 0.3,
    use_kalman: bool = False,
) -> List[Dict[str, Any]]:
    """Run YOLO detection on frames in `frames_dir` and save results.

    Args:
        frames_dir: directory of pre-extracted frames
        model_path: path to a YOLO weights file (e.g., 'yolov8n.pt' or path to a model)
        output_csv: CSV output path (set None to skip)
        output_json: JSON output path (set None to skip)
        annotated_dir: if provided, annotated images will be saved here
        batch_size: number of images to process per model call
        imgsz: inference image size
        conf: confidence threshold
        device: 'cpu' or 'cuda'
        target_class_names: list of class names to keep (e.g., ['ball'])
        target_class_ids: list of class ids to keep
        verbose: print progress
        enable_motion_prediction: if True, fill detection gaps with motion predictions
        max_gap_frames: maximum gap size (in frames) to fill with predictions
        prediction_confidence: confidence score assigned to predicted detections (0-1)
        use_kalman: use Kalman filtering for gaps > 3 frames (more accurate but slower)

    Returns:
        records: list of detection dictionaries (actual + predicted if enabled)
    """
    frames = get_image_paths(frames_dir)
    if len(frames) == 0:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")

    ensure_dir(annotated_dir)

    # Load model
    model = YOLO(model_path)

    # map target class names -> ids using model.names (dict)
    names_map = getattr(model, 'names', None) or {}
    if target_class_names and not target_class_ids:
        resolved = []
        for n in target_class_names:
            # try to find name by exact match
            found = False
            for cid, cname in names_map.items():
                if str(cname).lower() == str(n).lower():
                    resolved.append(int(cid))
                    found = True
                    break
            if not found:
                # warn but continue
                if verbose:
                    print(f"Warning: class name '{n}' not found in model.names; skipping it")
        target_class_ids = resolved

    total = len(frames)
    records: List[Dict[str, Any]] = []
    processed = 0

    iterator = range(0, total, batch_size)
    if _HAS_TQDM and verbose:
        iterator = tqdm(list(iterator), desc='Batches')

    for start in iterator:
        batch_paths = frames[start:start + batch_size]

        # Basic validation: ensure files readable by cv2
        valid_paths = []
        path_to_img = {}
        for p in batch_paths:
            try:
                img = cv2.imdecode(np_fromfile(p), cv2.IMREAD_COLOR) if False else cv2.imread(p)
                if img is None:
                    if verbose:
                        print(f"Warning: could not read image {p}; skipping")
                    continue
                valid_paths.append(p)
                path_to_img[p] = img
            except Exception:
                if verbose:
                    print(f"Warning: exception reading image {p}; skipping\n" + traceback.format_exc())

        if len(valid_paths) == 0:
            processed += len(batch_paths)
            # progress print
            if verbose:
                print(f"Processed {processed}/{total} frames (skipped unreadable files in this batch)")
            continue

        # Run model on batch (ultralytics can accept list of paths)
        try:
            results = model.predict(source=valid_paths, imgsz=imgsz, conf=conf, device=device, verbose=False)
        except Exception as e:
            # Try per-image fallback
            if verbose:
                print(f"Batch inference failed: {e}. Falling back to per-image inference.")
            results = []
            for p in valid_paths:
                try:
                    r = model.predict(source=p, imgsz=imgsz, conf=conf, device=device, verbose=False)
                    # predict returns a list of results even for a single image; extend accordingly
                    if isinstance(r, (list, tuple)):
                        results.extend(r)
                    else:
                        results.append(r)
                except Exception:
                    if verbose:
                        print(f"Warning: inference failed for {p}\n" + traceback.format_exc())

        # Parse results: results is iterable with element per image in valid_paths order
        for res, img_path in zip(results, valid_paths):
            frame_index = frames.index(img_path)
            frame_id = os.path.relpath(img_path, frames_dir)

            # ultralytics results: res.boxes with .xyxy, .conf, .cls
            boxes = []
            confs = []
            cls_names = []
            try:
                b_xyxy = getattr(res.boxes, 'xyxy', None)
                b_conf = getattr(res.boxes, 'conf', None)
                b_cls = getattr(res.boxes, 'cls', None)
                if b_xyxy is None or b_conf is None or b_cls is None:
                    # no detections
                    if verbose:
                        pass
                else:
                    # convert tensors to lists
                    xy = b_xyxy.cpu().numpy() if hasattr(b_xyxy, 'cpu') else b_xyxy.numpy()
                    cf = b_conf.cpu().numpy() if hasattr(b_conf, 'cpu') else b_conf.numpy()
                    cid = b_cls.cpu().numpy().astype(int) if hasattr(b_cls, 'cpu') else b_cls.numpy().astype(int)
                    for (x1, y1, x2, y2), cval, cidval in zip(xy, cf, cid):
                        if target_class_ids and len(target_class_ids) > 0 and int(cidval) not in target_class_ids:
                            continue
                        cname = names_map.get(int(cidval), str(int(cidval)))
                        boxes.append((float(x1), float(y1), float(x2), float(y2)))
                        confs.append(float(cval))
                        cls_names.append(cname)
            except Exception:
                if verbose:
                    print(f"Warning: error parsing results for {img_path}\n" + traceback.format_exc())

            if len(boxes) == 0:
                # no detections for this image
                # We still add a record optionally? Spec says handle gracefully -> skip adding detections but continue
                pass
            else:
                for (x1, y1, x2, y2), cval, cname, cidval in zip(boxes, confs, cls_names, [None]*len(boxes)):
                    record = {
                        'frame_index': frame_index,
                        'frame_id': frame_id,
                        'x_min': x1,
                        'y_min': y1,
                        'x_max': x2,
                        'y_max': y2,
                        'confidence': cval,
                        'class_id': None,
                        'class_name': cname,
                        'detection_type': 'detected',
                    }
                    records.append(record)

                # Save annotated image if requested
                if annotated_dir:
                    try:
                        # safe read (check cache first)
                        img = _safe_read(img_path, cache=path_to_img)
                        if img is None:
                            if verbose:
                                print(f"Warning: could not read image {img_path} for annotation; skipping annotation")
                        else:
                            ann_boxes = boxes
                            ann_confs = confs
                            ann_names = cls_names
                            annotated = draw_boxes_on_image(img.copy(), ann_boxes, ann_confs, ann_names)
                            out_name = os.path.join(annotated_dir, os.path.basename(img_path))
                            _safe_imwrite(out_name, annotated)
                    except Exception:
                        if verbose:
                            print(f"Warning: failed to write annotated image for {img_path}\n" + traceback.format_exc())

        processed += len(batch_paths)
        # Progress print
        if verbose and not _HAS_TQDM:
            print(f"Processed {min(processed, total)}/{total} frames")

    # Apply motion prediction if enabled
    if enable_motion_prediction and len(records) > 0:
        if verbose:
            print(f"\n[INFO] Applying motion prediction to fill detection gaps...")
        
        from detection.ball_tracker import fill_detection_gaps, filter_ball_detections
        
        # Filter to ball detections only if we have multiple classes
        ball_records = records
        if target_class_names:
            ball_records = filter_ball_detections(records, class_name=target_class_names[0] if target_class_names else 'sports ball')
        
        # Fill gaps
        original_count = len(ball_records)
        filled_records = fill_detection_gaps(
            ball_records,
            max_gap_frames=max_gap_frames,
            prediction_confidence=prediction_confidence,
            use_kalman=use_kalman
        )
        predicted_count = len(filled_records) - original_count
        
        if verbose:
            print(f"[INFO] Added {predicted_count} predicted detections to fill gaps")
        
        # Update records with frame_id for predicted detections
        frame_id_map = {r['frame_index']: r.get('frame_id', '') for r in records}
        for rec in filled_records:
            if 'frame_id' not in rec or not rec['frame_id']:
                # Try to infer frame_id from frame_index
                frame_idx = rec['frame_index']
                if frame_idx in frame_id_map:
                    rec['frame_id'] = frame_id_map[frame_idx]
                else:
                    # Use a placeholder or closest frame
                    rec['frame_id'] = f"frame_{frame_idx:06d}.jpg"
        
        # Re-annotate images with predicted detections if annotated_dir provided
        if annotated_dir and predicted_count > 0:
            if verbose:
                print(f"[INFO] Re-annotating images with predicted detections...")
            
            # Group detections by frame
            from detection.ball_tracker import group_detections_by_frame
            frame_detections = group_detections_by_frame(filled_records)
            
            for frame_idx, dets in frame_detections.items():
                # Only re-annotate frames that have predictions
                has_predictions = any(d.get('detection_type') == 'predicted' for d in dets)
                if not has_predictions:
                    continue
                
                # Find the original frame path
                frame_path = frames[frame_idx] if frame_idx < len(frames) else None
                if frame_path is None:
                    continue
                
                try:
                    img = cv2.imread(frame_path)
                    if img is None:
                        continue
                    
                    boxes = [(d['x_min'], d['y_min'], d['x_max'], d['y_max']) for d in dets]
                    confs = [d['confidence'] for d in dets]
                    names = [d['class_name'] for d in dets]
                    types = [d.get('detection_type', 'detected') for d in dets]
                    
                    annotated = draw_boxes_on_image(img.copy(), boxes, confs, names, detection_types=types)
                    out_name = os.path.join(annotated_dir, os.path.basename(frame_path))
                    _safe_imwrite(out_name, annotated)
                except Exception:
                    if verbose:
                        print(f"Warning: failed to re-annotate {frame_path}")
        
        records = filled_records

    # Save outputs
    write_outputs(records, output_csv, output_json)
    return records


def _get_model(model_path: str, device: str = 'cpu'):
    """Load and cache a YOLO model instance for given path+device."""
    key = f"{model_path}::device={device}"
    if not hasattr(_get_model, '_cache'):
        _get_model._cache = {}
    cache = _get_model._cache
    if key in cache:
        return cache[key]
    model = YOLO(model_path)
    cache[key] = model
    return model


def detect_ball_on_image(
    image,  # numpy array (BGR) or path string
    model_path: str = 'yolov8n.pt',
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = 'cpu',
    target_class_names: Optional[Sequence[str]] = None,
    target_class_ids: Optional[Sequence[int]] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Run YOLO on a single image (numpy array BGR or file path) and return detection records.

    Returns list of dicts with keys: x_min, y_min, x_max, y_max, confidence, class_id, class_name
    """
    # prepare image source
    src = image
    is_path = isinstance(image, str)
    if is_path:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        src = image

    model = _get_model(model_path, device=device)
    # resolve class ids from names if provided
    names_map = getattr(model, 'names', None) or {}
    if target_class_names and not target_class_ids:
        resolved = []
        for n in target_class_names:
            for cid, cname in names_map.items():
                if str(cname).lower() == str(n).lower():
                    resolved.append(int(cid))
                    break
        target_class_ids = resolved

    # run prediction
    try:
        results = model.predict(source=src, imgsz=imgsz, conf=conf, device=device, verbose=False)
    except Exception as e:
        if verbose:
            print(f"detect_ball_on_image: prediction failed: {e}")
        return []

    if not results:
        return []

    # results may be list-like; take first
    res = results[0]
    out: List[Dict[str, Any]] = []
    try:
        b_xyxy = getattr(res.boxes, 'xyxy', None)
        b_conf = getattr(res.boxes, 'conf', None)
        b_cls = getattr(res.boxes, 'cls', None)
        if b_xyxy is None or b_conf is None or b_cls is None:
            return []
        xy = b_xyxy.cpu().numpy() if hasattr(b_xyxy, 'cpu') else b_xyxy.numpy()
        cf = b_conf.cpu().numpy() if hasattr(b_conf, 'cpu') else b_conf.numpy()
        cid = b_cls.cpu().numpy().astype(int) if hasattr(b_cls, 'cpu') else b_cls.numpy().astype(int)
        for (x1, y1, x2, y2), cval, cidval in zip(xy, cf, cid):
            if target_class_ids and len(target_class_ids) > 0 and int(cidval) not in target_class_ids:
                continue
            cname = names_map.get(int(cidval), str(int(cidval)))
            out.append({
                'x_min': float(x1),
                'y_min': float(y1),
                'x_max': float(x2),
                'y_max': float(y2),
                'confidence': float(cval),
                'class_id': int(cidval),
                'class_name': cname,
            })
    except Exception:
        if verbose:
            print("detect_ball_on_image: error parsing results\n" + traceback.format_exc())
    return out


def detect_ball_on_image_adaptive(
    image,  # numpy array (BGR) or path string
    model_path: str = 'yolov8n.pt',
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = 'cpu',
    target_class_names: Optional[Sequence[str]] = None,
    target_class_ids: Optional[Sequence[int]] = None,
    enable_preprocessing: bool = True,
    target_brightness: float = 0.5,
    log_enhancements: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run YOLO with adaptive preprocessing on a single image.
    
    This function assesses image quality and applies preprocessing if needed,
    then runs detection with quality-adjusted confidence thresholds.
    
    Args:
        image: numpy array (BGR) or file path string
        model_path: path to YOLO weights
        imgsz: inference image size
        conf: base confidence threshold
        device: 'cpu' or 'cuda'
        target_class_names: list of class names to keep (e.g., ['ball'])
        target_class_ids: list of class ids to keep
        enable_preprocessing: enable/disable preprocessing
        target_brightness: target brightness for normalization (0-1)
        log_enhancements: whether to log enhancement operations
        verbose: print additional info
    
    Returns:
        Dictionary containing:
            - detections: List of detection dicts (x_min, y_min, x_max, y_max, confidence, class_id, class_name)
            - quality_score: Overall quality score (0-1)
            - was_preprocessed: Boolean indicating if preprocessing was applied
            - preprocessing_time_ms: Time spent on preprocessing
            - enhancements_applied: List of enhancement operations
            - confidence_threshold: Adjusted confidence threshold used
            - total_time_ms: Total processing time
    """
    try:
        from detection.frame_preprocessing import AdaptiveBallDetector
    except ImportError:
        if verbose:
            print("Warning: frame_preprocessing module not found. Install required dependencies or disable preprocessing.")
        # Fallback to standard detection
        detections = detect_ball_on_image(
            image, model_path, imgsz, conf, device,
            target_class_names, target_class_ids, verbose
        )
        return {
            'detections': detections,
            'quality_score': 0.0,
            'was_preprocessed': False,
            'preprocessing_time_ms': 0.0,
            'enhancements_applied': [],
            'confidence_threshold': conf,
            'total_time_ms': 0.0
        }
    
    # Load image if path provided
    frame = image
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        frame = cv2.imread(image)
        if frame is None:
            raise ValueError(f"Could not read image: {image}")
    
    # Load model
    model = _get_model(model_path, device=device)
    
    # Create adaptive detector
    detector = AdaptiveBallDetector(
        model=model,
        enable_preprocessing=enable_preprocessing,
        quality_threshold=0.6,
        base_confidence=conf,
        min_confidence=0.15,
        target_brightness=target_brightness,
        log_enhancements=log_enhancements
    )
    
    # Run adaptive detection
    frame_id = image if isinstance(image, str) else "frame"
    result = detector.detect(
        frame=frame,
        frame_id=frame_id,
        imgsz=imgsz,
        device=device
    )
    
    # Filter by target classes if specified
    names_map = getattr(model, 'names', None) or {}
    if target_class_names and not target_class_ids:
        resolved = []
        for n in target_class_names:
            for cid, cname in names_map.items():
                if str(cname).lower() == str(n).lower():
                    resolved.append(int(cid))
                    break
        target_class_ids = resolved
    
    # Format detections
    detections = []
    for box, conf_val, cls_id, cls_name in zip(
        result['boxes'], result['confidences'], 
        result['class_ids'], result['class_names']
    ):
        if target_class_ids and len(target_class_ids) > 0 and cls_id not in target_class_ids:
            continue
        
        detections.append({
            'x_min': float(box[0]),
            'y_min': float(box[1]),
            'x_max': float(box[2]),
            'y_max': float(box[3]),
            'confidence': float(conf_val),
            'class_id': int(cls_id),
            'class_name': cls_name,
        })
    
    # Return comprehensive result
    return {
        'detections': detections,
        'quality_score': result['quality_score'],
        'was_preprocessed': result['was_preprocessed'],
        'preprocessing_time_ms': result['preprocessing_time_ms'],
        'enhancements_applied': result['enhancements_applied'],
        'confidence_threshold': result['confidence_threshold'],
        'total_time_ms': result['total_time_ms']
    }


# Small helper: numpy file read fallback (handles long Windows paths using cv2.imdecode)
def np_fromfile(path):
    """Return numpy buffer suitable for cv2.imdecode (not used by default)."""
    import numpy as _np
    try:
        with open(path, 'rb') as f:
            file_bytes = f.read()
        return _np.frombuffer(file_bytes, _np.uint8)
    except Exception:
        return None


if __name__ == '__main__':
    frames_directory = 'test/images'  # update to your frames folder
    model_weights = 'yolov8n.pt'  # or path to your weights (yolov8n.pt included in repo)

    records = detect_balls(
        frames_dir=frames_directory,
        model_path=model_weights,
        output_csv='runs/detections_ball.csv',
        output_json='runs/detections_ball.json',
        annotated_dir='runs/annotated',
        batch_size=8,
        imgsz=640,
        conf=0.25,
        device='cpu',
        target_class_names=['ball'],
        verbose=True,
    )
    print(f"Written {len(records)} detection records")
