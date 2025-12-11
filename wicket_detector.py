# wicket_tracker_v5.py
import cv2
import numpy as np
from ultralytics import YOLO

class WicketTrackerV5:
    """
    Changes from v4:
     - If detector outputs a thinner (narrower) far-box, expand width to the RIGHT only.
     - Never reduce height from previously-known height; don't recompute height from width.
     - Near wicket not constrained by ratio; treat as visible portion only.
     - Conservative px-level width clamps; safe resync blending on reappearance.
    """

    def __init__(self,
                 model_path="weights/wicket_weights.pt",
                 min_det_conf=0.25,
                 smooth_top_alpha=0.2,
                 width_alpha=0.3,
                 width_px_cap=2,               # +/- pixels allowed per frame
                 initial_width_ratio=0.30,     # initial far width ≈ anchor_width * ratio
                 max_width_expand_factor=500,  # safety cap relative to detected width
                 max_dx_per_frame=10,
                 max_dy_per_frame=2,
                 anchor_miss_count=0,
                 max_anchor_miss=60,
                 last_anchor_det = None,
                 min_det_height_ratio_full=0.75):
        self.model = YOLO(model_path)
        self.min_det_conf = min_det_conf

        # smoothing / clamps
        self.smooth_top_alpha = smooth_top_alpha
        self.width_alpha = width_alpha
        self.width_px_cap = width_px_cap
        self.initial_width_ratio = initial_width_ratio
        self.max_width_expand_factor = max_width_expand_factor
        self.max_dx = max_dx_per_frame
        self.max_dy = max_dy_per_frame
        self.min_det_height_ratio_full = min_det_height_ratio_full

        self.last_anchor_det = last_anchor_det
        self.anchor_miss_count = anchor_miss_count
        self.max_anchor_miss = max_anchor_miss

        # state
        self.anchor_top_smoothed = None

        self.locked = False
        self.ref_anchor_width = None
        self.ref_offset_top = None

        # far state
        self.last_good_top = None
        self.last_known_top = None
        self.prev_far_width = None
        self.prev_far_height = None
        self.prev_far_area = None

    # utils
    def _top_left(self, box):
        x, y, w, h = box
        return np.array([float(x), float(y)], dtype=float)

    def _top_center(self, box):
        x, y, w, h = box
        return np.array([x + w/2.0, float(y)], dtype=float)

    def _smooth(self, prev, new, alpha):
        if prev is None:
            return new.copy()
        return prev * (1.0 - alpha) + new * alpha

    def _clamp_step(self, prev, target, max_dx, max_dy):
        if prev is None:
            return target.copy()
        dx = target[0] - prev[0]
        dy = target[1] - prev[1]
        dx_c = np.clip(dx, -max_dx, max_dx)
        dy_c = np.clip(dy, -max_dy, max_dy)
        return np.array([prev[0] + dx_c, prev[1] + dy_c], dtype=float)

    def _is_full_detection(self, det_h, expected_h):
        if expected_h is None:
            return det_h > 12
        return det_h >= expected_h * self.min_det_height_ratio_full

    # main
    def detect_and_track(self, frame, conf=0.25):
        h_img, w_img = frame.shape[:2]

        # run YOLO
        results = self.model.predict(frame, conf=conf, verbose=False)
        dets = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w = x2 - x1
                h = y2 - y1
                c = float(box.conf[0])
                dets.append({"box": [x1, y1, w, h], "conf": c, "area": w*h})

        if not dets:
            return frame, []

        # choose anchor: largest in lower half (anchor box kept unchanged)
        candidates = [d for d in dets if (d["box"][1] + d["box"][3]/2.0) > h_img * 0.45]
        candidates.sort(key=lambda d: d["area"], reverse=True)
        current_anchor = candidates[0] if candidates else None

        anchor = None
        if current_anchor:
            self.last_anchor_det = current_anchor
            self.anchor_miss_count = 0
            anchor = current_anchor
        elif self.last_anchor_det is not None and self.anchor_miss_count < self.max_anchor_miss:
            anchor = self.last_anchor_det
            self.anchor_miss_count += 1
            ax, ay, aw, ah = anchor["box"]
            cv2.rectangle(frame, (ax, ay), (ax+aw, ay+ah), (100,100,100), 2)

        if anchor is None:
            return frame, []
        # smooth anchor top only (do NOT change anchor box)
        # anchor_top_raw = self._top_center(anchor["box"])
        anchor_top_raw = self._top_left(anchor["box"])
        if self.anchor_top_smoothed is None:
            self.anchor_top_smoothed = anchor_top_raw
        else:
            # Calculate how much the anchor moved this frame
            dist = np.linalg.norm(anchor_top_raw - self.anchor_top_smoothed)
            
            # --- NEW: DEADBAND ---
            if dist < 3.0: 
                # Movement is too small; ignore it to prevent jitter transfer
                # Keep self.anchor_top_smoothed exactly as it was
                pass 
            else:
                # Movement is real; apply smoothing
                self.anchor_top_smoothed = self._smooth(self.anchor_top_smoothed, anchor_top_raw, self.smooth_top_alpha)
        
        # draw anchor (use YOLO box)
        ax, ay, aw, ah = anchor["box"]
        cv2.rectangle(frame, (ax, ay), (ax+aw, ay+ah), (0,255,0), 2)
        cv2.putText(frame, "Wicket_Near", (ax, ay-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # find far candidates (above anchor, low overlap)
        far_candidates = []
        for d in dets:
            # tx, ty = self._top_center(d["box"])
            tx, ty = self._top_left(d["box"])
            if ty < self.anchor_top_smoothed[1]:
                # simple IoU overlap check to avoid anchor duplicate
                bx, by, bw, bh = d["box"]
                ax2, ay2, aw2, ah2 = anchor["box"]
                xA = max(bx, ax2); yA = max(by, ay2)
                xB = min(bx + bw, ax2 + aw2); yB = min(by + bh, ay2 + ah2)
                inter = max(0, xB - xA) * max(0, yB - yA)
                iou = inter / (bw*bh + aw2*ah2 - inter + 1e-6)
                if iou < 0.45:
                    far_candidates.append(d)
        far_candidates.sort(key=lambda d: d["area"], reverse=True)
        far = far_candidates[0] if far_candidates else None

        # Prepare outputs
        final_dets = [{"label": "Wicket_Near", "box": anchor["box"], "conf": anchor["conf"]}]

        # expected height if we have prev height
        expected_far_h = int(self.prev_far_height) if self.prev_far_height is not None else None

        # Decide if far detection is "full"
        far_is_full = False
        if far and far["conf"] >= self.min_det_conf:
            _, _, fw, fh = far["box"]
            if self._is_full_detection(fh, expected_far_h):
                # simple ar sanity check
                ar = fh / (fw + 1e-9) if fw > 0 else 0
                if 1.0 < ar < 6.0:
                    far_is_full = True

        # If full detection => update last_good_top and widths/heights conservatively
        if far_is_full:
            fx, fy, fw, fh = far["box"]
            # far_top = self._top_center(far["box"])
            far_top = self._top_left(far["box"])

            if self.locked and far_is_full:
                new_top = self._top_left(far["box"])
                # Calculate distance from our steady state
                dist = np.linalg.norm(new_top - self.last_known_top)
                
                # If it jumped more than 20px, it's likely a pad or fielder leg
                if dist > 20: 
                    far_is_full = False

            # initialize lock if needed
            if not self.locked:
                self.locked = True
                self.ref_anchor_width = anchor["box"][2]
                self.ref_offset_top = far_top - self.anchor_top_smoothed
                # set prev widths/heights from detection but respect tiny px clamp
                self.prev_far_width = int(fw)
                self.prev_far_height = int(fh)
                self.last_good_top = far_top.copy()
                self.last_known_top = far_top.copy()
                self.prev_far_area = fw * fh
            else:
                # update last_good_top (blend based on mismatch)
                if self.last_good_top is None:
                    self.last_good_top = far_top.copy()
                else:
                    mismatch = np.linalg.norm(far_top - self.last_good_top)
                    alpha = 0.28 if mismatch < 10 else 0.12
                    self.last_good_top = self._smooth(self.last_good_top, far_top, alpha)

                # Width handling: DO NOT ALLOW shrink. If detector narrower, expand to right to prev width.
                if self.prev_far_width is None:
                    # estimate from anchor if we don't have a prev
                    est_w = int(anchor["box"][2] * self.initial_width_ratio)
                    adjusted_w = max(fw, est_w)
                else:
                    adjusted_w = max(fw, self.prev_far_width)

                # safety cap relative to detector to avoid extreme expansion
                adjusted_w = min(adjusted_w, int(fw * self.max_width_expand_factor))

                # apply px-level smoothing/clamp
                if self.prev_far_width is None:
                    width_used = int(adjusted_w)
                else:
                    width_used = int(self.prev_far_width * (1.0 - self.width_alpha) + adjusted_w * self.width_alpha)
                    width_used = int(np.clip(width_used,
                                             self.prev_far_width - self.width_px_cap,
                                             self.prev_far_width + self.width_px_cap))

                # Height: DO NOT let height drop — keep prev height if exists
                if self.prev_far_height is None:
                    height_used = int(fh)
                else:
                    # prefer previous height (no shrinking). If detector has larger height, accept small increases
                    if fh > self.prev_far_height:
                        height_used = int(self.prev_far_height * (1.0 - 0.12) + fh * 0.12)
                    else:
                        height_used = int(self.prev_far_height)

                # update prevs
                self.prev_far_width = width_used
                self.prev_far_height = height_used
                self.prev_far_area = width_used * height_used

                # last_known_top softly follows last_good_top
                if self.last_known_top is None:
                    self.last_known_top = self.last_good_top.copy()
                else:
                    self.last_known_top = self._smooth(self.last_known_top, self.last_good_top, 0.25)

            # Build box: **expand to the RIGHT only** so left x stays the same
            left_x = fx
            width_final = int(self.prev_far_width)
            height_final = int(self.prev_far_height)
            # ensure doesn't exceed image bounds
            if left_x + width_final > w_img:
                width_final = max(4, w_img - left_x)
            top_for_draw = self.last_known_top if self.last_known_top is not None else far_top
            box_x = int(left_x)
            box_y = int(top_for_draw[1])
            far_box = [box_x, box_y, width_final, height_final]

            cv2.rectangle(frame, (far_box[0], far_box[1]), (far_box[0]+far_box[2], far_box[1]+far_box[3]), (0,128,255), 2)
            cv2.putText(frame, "Wicket_Far (Det)", (far_box[0], far_box[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,255), 2)
            final_dets.append({"label": "Wicket_Far", "box": far_box, "conf": 0.95})
            return frame, final_dets

        # If not full detection:
        # If not locked -> we can't reliably predict; but we still expand thin detections to right if present
        if not self.locked:
            # if a partial far detection exists, expand to right heuristically to improve starting frames
            if far and far["conf"] >= self.min_det_conf:
                fx, fy, fw, fh = far["box"]
                # estimate width from anchor if no prev
                est_w = int(anchor["box"][2] * self.initial_width_ratio)
                adjusted_w = max(fw, est_w)
                adjusted_w = min(adjusted_w, int(fw * self.max_width_expand_factor))
                # keep height as detection (do NOT change)
                height_used = int(fh)
                left_x = fx
                # apply px clamp relative to previous if exists
                if self.prev_far_width is not None:
                    adjusted_w = int(np.clip(adjusted_w,
                                             self.prev_far_width - self.width_px_cap,
                                             self.prev_far_width + self.width_px_cap))
                # final
                if left_x + adjusted_w > w_img:
                    adjusted_w = max(4, w_img - left_x)
                far_box = [int(left_x), int(fy), int(adjusted_w), int(height_used)]
                # update prevs
                self.prev_far_width = far_box[2]
                self.prev_far_height = far_box[3]
                # self.last_known_top = self._top_center(far_box)
                self.last_known_top = self._top_left(far_box)

                cv2.rectangle(frame, (far_box[0], far_box[1]), (far_box[0]+far_box[2], far_box[1]+far_box[3]), (0,128,200), 2)
                cv2.putText(frame, "Wicket_Far (PartialDet)", (far_box[0], far_box[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,200), 2)
                final_dets.append({"label": "Wicket_Far", "box": far_box, "conf": 0.8})
                return frame, final_dets
            # else nothing reliable
            return frame, final_dets

        # locked but far not fully visible => predict using previous width/height and anchor offset
        # compute tiny scale from anchor (mostly 1.0)
        scale = float(anchor["box"][2]) / (self.ref_anchor_width + 1e-9)
        # scale = 1.0 + (scale - 1.0) * 0.12

        # width: reuse prev_far_width and allow only tiny px change
        if self.prev_far_width is None:
            pred_w = int(self.ref_anchor_width * self.initial_width_ratio)
        else:
            pred_w = int(self.prev_far_width * (1.0 + (scale - 1.0) * 0.05))
            pred_w = int(np.clip(pred_w, self.prev_far_width - self.width_px_cap, self.prev_far_width + self.width_px_cap))

        # height: preserve previous height (do not recompute)
        if self.prev_far_height is None:
            pred_h = int(pred_w * 3.11)  # fallback
        else:
            pred_h = int(self.prev_far_height)

        # predicted top
        predicted_top = self.anchor_top_smoothed + self.ref_offset_top * scale

        # clamp predicted top relative to last_good_top/last_known_top to avoid sinking
        anchor_for_clamp = self.last_good_top if self.last_good_top is not None else (self.last_known_top if self.last_known_top is not None else predicted_top)
        clamped_top = self._clamp_step(anchor_for_clamp, predicted_top, self.max_dx, self.max_dy)

        # smooth last_known_top
        if self.last_known_top is None:
            self.last_known_top = clamped_top.copy()
        else:
            self.last_known_top = self._smooth(self.last_known_top, clamped_top, 0.18)

        # update prevs
        self.prev_far_width = int(pred_w)
        self.prev_far_height = int(pred_h)

        # fx = int(self.last_known_top[0] - pred_w/2.0)
        fx = int(self.last_known_top[0])
        fy = int(self.last_known_top[1])

        # ensure within image
        if fx < 0:
            # since user said batsman will be on the right, expand to right but not left
            fx = 0
        if fx + pred_w > w_img:
            pred_w = max(4, w_img - fx)

        # clamp bottom to image
        bottom_limit = int(h_img * 0.98)
        if fy + pred_h > bottom_limit:
            pred_h = max(4, bottom_limit - fy)
            # keep width consistent (we won't recompute height from width, so scale width proportionally small)
            if pred_h < self.prev_far_height:
                # avoid changing width; instead cap height and keep width same (user insisted no height tinkering)
                pass

        far_box = [fx, fy, int(pred_w), int(pred_h)]
        cv2.rectangle(frame, (far_box[0], far_box[1]), (far_box[0]+far_box[2], far_box[1]+far_box[3]), (0,255,255), 2)
        cv2.putText(frame, "Wicket_Far (Pred)", (far_box[0], far_box[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        final_dets.append({"label": "Wicket_Far", "box": far_box, "conf": 1.0})
        return frame, final_dets


# singletons
_tracker_v5 = None
def detect_wicket(frame, conf=0.25):
    global _tracker_v5
    if _tracker_v5 is None:
        _tracker_v5 = WicketTrackerV5(min_det_conf=conf)
    return _tracker_v5.detect_and_track(frame, conf)
