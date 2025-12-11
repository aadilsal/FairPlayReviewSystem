import numpy as np

# -----------------------------
# Geometry helpers
# -----------------------------

def iou_xyxy(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    denom = areaA + areaB - inter
    return 0.0 if denom <= 0 else inter / denom


def center_xywh(box):
    x, y, w, h = box
    return np.array([x + w / 2.0, y + h / 2.0])


# -----------------------------
# Batsman Finder (Fixed: No Averaging)
# -----------------------------

class BatsmanFinder:

    SEARCH = 0
    CONFIRMED = 1

    def __init__(
        self,
        iou_thresh=0.05,
        consec_required=3,
    ):
        self.iou_thresh = iou_thresh
        self.consec_required = consec_required

        self.state = self.SEARCH
        self.candidate_box = None   # xywh
        self.consec_count = 0
        self.confirmed_bbox = None

    def _reset(self):
        if self.consec_count > 0:
            print(f"[DEBUG] RESET triggered. Counter dropped from {self.consec_count} to 0.")
        self.candidate_box = None
        self.consec_count = 0

    def process_frame(self, frame, persons, bats, frame_idx=None):
        """
        frame: The image frame
        persons: List of person detections passed from the pipeline
        bats: List of bat detections passed from the pipeline
        frame_idx: Index for metadata
        """

        meta = {
            "frame_index": frame_idx,
            "state": "CONFIRMED" if self.state == self.CONFIRMED else "SEARCH",
            "consec_count": int(self.consec_count),
            "batsman_confirmed": (self.state == self.CONFIRMED),
            "persons": len(persons),
            "bats": len(bats),
            "best_iou": 0.0,
        }

        # ✅ HARD LOCK AFTER CONFIRMATION
        if self.state == self.CONFIRMED:
            meta["batsman_bbox"] = list(map(int, self.confirmed_bbox))
            return frame, meta

        # --------------------------------------------------
        # 1) SEARCH — lock best candidate ONCE
        # --------------------------------------------------
        if self.candidate_box is None:
            best_iou = 0.0
            best_person = None

            for b in bats:
                bx, by, bw, bh = b["box"]
                b_xyxy = (bx, by, bx + bw, by + bh)

                for p in persons:
                    px, py, pw, ph, _ = p
                    p_xyxy = (px, py, px + pw, py + ph)
                    i = iou_xyxy(b_xyxy, p_xyxy)
                    if i > best_iou:
                        best_iou = i
                        best_person = (px, py, pw, ph)

            meta["best_iou"] = float(best_iou)

            if best_person and best_iou >= self.iou_thresh:
                self.candidate_box = best_person
                self.consec_count = 1
                print(f"[DEBUG] Frame {frame_idx}: FOUND candidate! IoU={best_iou:.4f} | Count=1")
            else:
                self._reset()

            return frame, meta

        # --------------------------------------------------
        # 2) VERIFY — same candidate only
        # --------------------------------------------------
        
        # Step A: Find the candidate in the CURRENT frame's person list
        cx, cy, cw, ch = self.candidate_box
        old_c_xyxy = (cx, cy, cx + cw, cy + ch)
        
        matched_person = None
        best_person_iou = 0.0
        
        for p in persons:
            px, py, pw, ph, _ = p
            p_xyxy = (px, py, px + pw, py + ph)
            i = iou_xyxy(old_c_xyxy, p_xyxy)
            if i > best_person_iou:
                best_person_iou = i
                matched_person = (px, py, pw, ph)
        
        if matched_person is None or best_person_iou < 0.1:
            print(f"[DEBUG] Frame {frame_idx}: Lost candidate person (overlap={best_person_iou:.2f}). Resetting.")
            self._reset()
            return frame, meta

        # ✅ Update to NEW position
        self.candidate_box = matched_person
        px, py, pw, ph = self.candidate_box
        p_xyxy = (px, py, px + pw, py + ph)

        # Step B: Check bat overlap with NEW person box
        best_bat_iou = 0.0
        for b in bats:
            bx, by, bw, bh = b["box"]
            b_xyxy = (bx, by, bx + bw, by + bh)
            i = iou_xyxy(b_xyxy, p_xyxy)
            best_bat_iou = max(best_bat_iou, i)

        meta["best_iou"] = float(best_bat_iou)

        if best_bat_iou >= self.iou_thresh:
            self.consec_count += 1
            print(f"[DEBUG] Frame {frame_idx}: VERIFYING... IoU={best_bat_iou:.4f} | Count={self.consec_count}/{self.consec_required}")
        else:
            print(f"[DEBUG] Frame {frame_idx}: FAILED bat verification. IoU={best_bat_iou:.4f} < {self.iou_thresh}. Resetting.")
            self._reset()
            return frame, meta

        meta["consec_count"] = int(self.consec_count)

        # --------------------------------------------------
        # 3) CONFIRMATION (Using EXACT current box)
        # --------------------------------------------------
        if self.consec_count >= self.consec_required:
            print(f"[DEBUG] Frame {frame_idx}: CONFIRMED! Reached {self.consec_count} frames.")
            
            # ✅ FIX: Use self.candidate_box directly (The latest tight YOLO detection)
            # No averaging, no "mean_center" lag.
            self.state = self.CONFIRMED
            self.confirmed_bbox = self.candidate_box 

            meta["state"] = "CONFIRMED"
            meta["batsman_confirmed"] = True
            meta["batsman_bbox"] = list(map(int, self.confirmed_bbox))

        return frame, meta