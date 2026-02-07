import cv2
import numpy as np

DECISION_OUT = "OUT"
DECISION_NOT_OUT = "NOT OUT"
DECISION_UMPIRES_CALL = "UMPIRE'S CALL"
DECISION_INCONCLUSIVE = "INCONCLUSIVE"
DECISION_NO_DECISION = "NO DECISION"


def _point_in_polygon(point, polygon):
    if point is None or not polygon:
        return False
    pts = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0


def compute_lbw_decision(impact_point, pitch_model, would_hit_stumps, confidence,
                         conf_threshold=0.75, min_confidence=0.55):
    if impact_point is None or pitch_model is None or would_hit_stumps is None:
        return DECISION_INCONCLUSIVE, "missing_inputs"
    if confidence is None or confidence < min_confidence:
        return DECISION_INCONCLUSIVE, "low_confidence"

    polygon = pitch_model.get("polygon")
    if polygon and not _point_in_polygon(impact_point, polygon):
        return DECISION_NOT_OUT, "impact_outside_pitch"

    if confidence < conf_threshold:
        return DECISION_UMPIRES_CALL, "borderline_confidence"

    if would_hit_stumps:
        return DECISION_OUT, "predicted_hit"
    return DECISION_NOT_OUT, "predicted_miss"
