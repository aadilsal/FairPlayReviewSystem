"""Coordinate system (image-space, pixels):
- Origin at top-left.
- +x to the right, +y down.
- Gravity acts in +y.
- Delivery direction (bowler -> batsman) is +x by convention.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PitchPlaneEstimator:
    def __init__(
        self,
        warmup_frames=90,
        resize_width=360,
        variance_threshold=12.0,
        min_static_ratio=0.45,
        prefer_lower_half=0.55,
        texture_threshold=18.0,
        edge_low=40,
        edge_high=140,
        hough_threshold=60,
        min_line_length=80,
        max_line_gap=25,
        min_region_area_ratio=0.08,
        row_fill_threshold=0.35,
        min_line_separation_ratio=0.2,
        min_confidence=0.35
    ):
        self.warmup_frames = warmup_frames
        self.resize_width = resize_width
        self.variance_threshold = variance_threshold
        self.min_static_ratio = min_static_ratio
        self.prefer_lower_half = prefer_lower_half
        self.texture_threshold = texture_threshold
        self.edge_low = edge_low
        self.edge_high = edge_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.min_region_area_ratio = min_region_area_ratio
        self.row_fill_threshold = row_fill_threshold
        self.min_line_separation_ratio = min_line_separation_ratio
        self.min_confidence = min_confidence
        self._frames = []
        self._frame_color = None
        self._pitch_y = None
        self._ready = False
        self._failed = False
        self._scale = 1.0
        self._model = None
        self._logged_start = False
        self._logged_ready = False
        self._logged_failed = False

    def add_frame(self, frame):
        if self._ready or self._failed:
            return self._pitch_y
        if not self._logged_start:
            logger.info("[Pitch] Warmup started: collecting %d frames", self.warmup_frames)
            self._logged_start = True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if self.resize_width and w > self.resize_width:
            self._scale = float(self.resize_width) / float(w)
            gray = cv2.resize(gray, (self.resize_width, int(h * self._scale)))
        else:
            self._scale = 1.0
        self._frames.append(gray)
        self._frame_color = frame
        if len(self._frames) < self.warmup_frames:
            return None
        model = self._infer_pitch_model()
        if model is None:
            self._failed = True
            if not self._logged_failed:
                logger.warning("[Pitch] Inference failed: low confidence or insufficient geometry")
                self._logged_failed = True
            return None
        self._model = model
        self._pitch_y = int(model["bottom_y"])
        self._ready = True
        if not self._logged_ready:
            logger.info("[Pitch] Inference ready: bottom_y=%d confidence=%.2f", self._pitch_y, model.get("confidence", 0.0))
            self._logged_ready = True
        self._frames = []
        return self._pitch_y

    def is_ready(self):
        return self._ready

    def is_failed(self):
        return self._failed

    def get_pitch_y(self):
        return self._pitch_y

    def get_pitch_y_at_x(self, x):
        if not self._model:
            return None
        return float(self._model["bottom_y"])

    def get_model(self):
        return self._model

    def get_pitch_polygon(self):
        if not self._model:
            return None
        return self._model.get("polygon")

    def _infer_pitch_model(self):
        stack = np.stack(self._frames, axis=0).astype(np.float32)
        var = np.var(stack, axis=0)
        static_mask = var < self.variance_threshold
        row_scores = static_mask.mean(axis=1)
        h_static = row_scores.shape[0]
        start = int(h_static * self.prefer_lower_half)
        if start >= h_static - 1:
            start = max(0, h_static // 2)
        if row_scores[start:].max() < self.min_static_ratio:
            return None
        median_gray = np.median(stack, axis=0).astype(np.uint8)
        lap = cv2.Laplacian(median_gray, cv2.CV_32F)
        lap = np.abs(lap)
        texture_mask = lap < self.texture_threshold
        candidate_mask = (static_mask & texture_mask).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        h, w = candidate_mask.shape[:2]
        scale = self._scale if self._scale > 0 else 1.0
        full_w = int(round(w / scale))
        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < (h * w * self.min_region_area_ratio):
            return None
        mask = np.zeros_like(candidate_mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)

        edges = cv2.Canny(median_gray, self.edge_low, self.edge_high)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        if lines is None:
            return None

        left_pts = []
        right_pts = []
        y_bottom = float(h - 1)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dy) < abs(dx) * 0.5:
                continue
            length = np.hypot(dx, dy)
            if length < self.min_line_length:
                continue
            if abs(dy) < 1.0:
                continue
            t = (y_bottom - y1) / dy
            x_at_bottom = x1 + dx * t
            if x_at_bottom < 0 or x_at_bottom >= w:
                continue
            if x_at_bottom < w * 0.5:
                left_pts.extend([(x1, y1), (x2, y2)])
            else:
                right_pts.extend([(x1, y1), (x2, y2)])

        if len(left_pts) < 4 or len(right_pts) < 4:
            return None

        left_pts = np.array(left_pts, dtype=np.float32)
        right_pts = np.array(right_pts, dtype=np.float32)
        left_fit = np.polyfit(left_pts[:, 1], left_pts[:, 0], 1)
        right_fit = np.polyfit(right_pts[:, 1], right_pts[:, 0], 1)
        left_a, left_b = float(left_fit[0]), float(left_fit[1])
        right_a, right_b = float(right_fit[0]), float(right_fit[1])

        y_bottom_full = y_bottom / scale
        left_b = left_b / scale
        right_b = right_b / scale

        x_left_bottom = left_a * y_bottom_full + left_b
        x_right_bottom = right_a * y_bottom_full + right_b
        if x_right_bottom - x_left_bottom < full_w * self.min_line_separation_ratio:
            return None

        valid_rows = []
        for y in range(0, h, 2):
            x_left = left_a * y + left_b
            x_right = right_a * y + right_b
            if x_right <= x_left:
                continue
            x_l = int(max(0, min(w - 1, x_left)))
            x_r = int(max(0, min(w - 1, x_right)))
            if x_r - x_l < 10:
                continue
            row_slice = mask[y, x_l:x_r]
            if row_slice.size == 0:
                continue
            if (row_slice > 0).mean() >= self.row_fill_threshold:
                valid_rows.append(y)

        if len(valid_rows) < 10:
            return None

        y_top = float(min(valid_rows))
        y_bottom = float(max(valid_rows))
        if y_bottom <= y_top + 10:
            return None

        mask_ratio = area / float(h * w)
        line_count = float(len(left_pts) + len(right_pts))
        confidence = min(1.0, mask_ratio / 0.2) * min(1.0, line_count / 24.0)
        if confidence < self.min_confidence:
            return None

        y_top_full = y_top / self._scale
        y_bottom_full = y_bottom / self._scale
        x_left_top = left_a * y_top_full + left_b
        x_right_top = right_a * y_top_full + right_b
        x_left_bottom = left_a * y_bottom_full + left_b
        x_right_bottom = right_a * y_bottom_full + right_b
        polygon = [
            (float(x_left_top), float(y_top_full)),
            (float(x_right_top), float(y_top_full)),
            (float(x_right_bottom), float(y_bottom_full)),
            (float(x_left_bottom), float(y_bottom_full))
        ]

        model = {
            "left_line": (left_a, left_b),
            "right_line": (right_a, right_b),
            "top_y": y_top_full,
            "bottom_y": y_bottom_full,
            "polygon": polygon,
            "confidence": confidence
        }
        return model
