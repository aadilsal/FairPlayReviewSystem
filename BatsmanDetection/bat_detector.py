from ultralytics import YOLO
import cv2
import logging
import time

logger = logging.getLogger(__name__)

# ── Model control ──────────────────────────────────────────────────────────────
# Set True to re-enable primary model for diagnosis
USE_PRIMARY_MODEL = True

# Set True to temporarily run BOTH models and compare output (diagnosis mode)
DIAGNOSTIC_MODE = False

# Set True to use fallback model, False to use primary model only
USE_FALLBACK_MODEL = False

primary_bat_model = YOLO('weights/bat_finetune_v1_best.pt') if USE_PRIMARY_MODEL else None
fallback_bat_model = YOLO('weights/Bat_detection2/weights/best.pt') if USE_FALLBACK_MODEL else None


def _collect_center_filtered_detections(results, x_min, x_max, model_name="Model"):
    detections = []
    all_detections = []

    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy[:4])
            conf_score = float(box.conf[0])
            cls_id = int(box.cls[0])          # ← ADD: log class id for diagnosis
            det_center_x = (x1 + x2) / 2

            all_detections.append({
                "x_center": det_center_x,
                "conf": conf_score,
                "cls_id": cls_id,
                "in_zone": x_min < det_center_x < x_max
            })

            if x_min < det_center_x < x_max:
                detections.append({
                    "label": "Bat",
                    "conf": round(conf_score, 4),
                    "box": [x1, y1, x2 - x1, y2 - y1]
                })

    raw_count = len(all_detections)
    zone_info = ""
    if raw_count > 0:
        in_zone = [d for d in all_detections if d["in_zone"]]
        out_zone = [d for d in all_detections if not d["in_zone"]]
        zone_info = f"RAW:{raw_count} In-zone:{len(in_zone)} Out-zone:{len(out_zone)}"
        if out_zone:
            details = [f"x={d['x_center']:.0f} cls={d['cls_id']} conf={d['conf']:.3f}"
                       for d in out_zone]
            zone_info += f" | Rejected: {', '.join(details)}"
    else:
        zone_info = "RAW:0 — model returned no boxes at all"   # ← BUG 3 FIX

    return detections, raw_count, len(all_detections) - len(detections), zone_info


def detect_bat(frame, conf=0.25, center_fraction=0.6):   # ← BUG 2 FIX: 0.25→0.6
    height, width = frame.shape[:2]
    zone_width = width * center_fraction
    x_min = (width - zone_width) / 2
    x_max = (width + zone_width) / 2

    detections = []

    # ── Primary model ──────────────────────────────────────────────────────────
    if USE_PRIMARY_MODEL and primary_bat_model is not None:
        start = time.time()
        # Run at near-zero conf first in diagnostic mode to see raw output
        run_conf = 0.01 if DIAGNOSTIC_MODE else conf
        primary_results = primary_bat_model.predict(frame, conf=run_conf, verbose=False)
        primary_time = (time.time() - start) * 1000

        detections, raw, filtered, zone_info = _collect_center_filtered_detections(
            primary_results, x_min, x_max, "PRIMARY"
        )

        if DIAGNOSTIC_MODE:
            # Log model class names once for diagnosis
            logger.info(f"[PRIMARY] model.names = {primary_bat_model.names}")

        if detections:
            avg_conf = sum(d['conf'] for d in detections) / len(detections)
            logger.info(f"[PRIMARY] ✓ {len(detections)} bat(s) | avg_conf={avg_conf:.3f} | {zone_info} | {primary_time:.0f}ms")
            if not DIAGNOSTIC_MODE:
                return frame, detections
            # In diagnostic mode, fall through to also run fallback for comparison
        else:
            # BUG 3 FIX: always log, even when raw == 0
            level = logger.warning if raw == 0 else logger.info
            level(f"[PRIMARY] ✗ No center-zone bats | {zone_info} | {primary_time:.0f}ms")

    # ── Fallback model ─────────────────────────────────────────────────────────
    if not USE_FALLBACK_MODEL:
        logger.debug("[FALLBACK] DISABLED (USE_FALLBACK_MODEL=False) - Primary model is the only detector")
        return frame, detections
    
    start = time.time()
    fallback_results = fallback_bat_model.predict(frame, conf=conf, verbose=False)
    fallback_time = (time.time() - start) * 1000

    fallback_detections, raw, filtered, zone_info = _collect_center_filtered_detections(
        fallback_results, x_min, x_max, "FALLBACK"
    )

    if DIAGNOSTIC_MODE and USE_PRIMARY_MODEL:
        logger.info(f"[FALLBACK] model.names = {fallback_bat_model.names}")

    if fallback_detections:
        avg_conf = sum(d['conf'] for d in fallback_detections) / len(fallback_detections)
        logger.warning(f"[FALLBACK] ✓ {len(fallback_detections)} bat(s) | avg_conf={avg_conf:.3f} | {zone_info} | {fallback_time:.0f}ms")
    else:
        logger.info(f"[FALLBACK] ✗ No center-zone bats | {zone_info} | {fallback_time:.0f}ms")

    # In diagnostic mode, prefer fallback result for pipeline output
    return frame, fallback_detections  # BUG 4 FIX: single return