"""Small harness to compare raw vs preprocessed detection on sample frames."""
import os
import logging
import cv2
from preprocessing import estimate_blur, preprocess_frame
from ball_detector import get_yolo_detector

logger = logging.getLogger(__name__)


def test_on_frames(frame_paths, weights_path=None, imgsz=640):
    detector = get_yolo_detector(weights_path)
    results = []
    for p in frame_paths:
        img = cv2.imread(p)
        if img is None:
            logger.error(f"Could not load {p}")
            continue
        blur = estimate_blur(img)
        # raw detections
        raw = detector.detect(img, imgsz=imgsz)
        # preprocessed
        proc, info = preprocess_frame(img)
        enhanced = detector.detect(proc, imgsz=imgsz)
        result = {
            'frame': p,
            'blur': blur,
            'raw_count': len(raw),
            'enhanced_count': len(enhanced),
            'improvement': len(enhanced) - len(raw),
        }
        results.append(result)
        logger.info(f"{os.path.basename(p)}: blur={blur:.1f} raw={len(raw)} enhanced={len(enhanced)}")
    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sample_dir = os.path.join('test_frames')
    if not os.path.exists(sample_dir):
        print('Place sample frames under test_frames/ and re-run')
    else:
        fps = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.png'))]
        if not fps:
            print('No sample frames found in test_frames/')
        else:
            res = test_on_frames(fps)
            total_improve = sum(r['improvement'] for r in res)
            print(f"Total improvement across samples: {total_improve}")
