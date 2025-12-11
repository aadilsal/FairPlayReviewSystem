# file: Batsman_finder.py
import numpy as np
from person_detector import detect_persons
from bat_detector import detect_bat

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
# Batsman Finder (STRICT FSM)
# -----------------------------

class BatsmanFinder:

    SEARCH = 0
    CONFIRMED = 1

    def __init__(
        self,
        person_conf=0.5,
        bat_conf=0.3,
        iou_thresh=0.1,
        consec_required=3,
    ):
        self.person_conf = person_conf
        self.bat_conf = bat_conf
        self.iou_thresh = iou_thresh
        self.consec_required = consec_required

        self.state = self.SEARCH
        self.candidate_box = None   # xywh
        self.consec_count = 0
        self.center_history = []
        self.confirmed_bbox = None

    def _reset(self):
        self.candidate_box = None
        self.consec_count = 0
        self.center_history = []

    def process_frame(self, frame, frame_idx=None):

        frame, persons = detect_persons(frame, person_conf=self.person_conf)
        frame, bats = detect_bat(frame, conf=self.bat_conf)

        meta = {
            "frame_index": frame_idx,
            "state": "CONFIRMED" if self.state == self.CONFIRMED else "SEARCH",
            "consec_count": int(self.consec_count),
            "batsman_confirmed": (self.state == self.CONFIRMED),
            "persons": persons,
            "bats": bats,
            "num_persons": len(persons),
            "num_bats": len(bats),
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
                self.center_history = [center_xywh(best_person)]
            else:
                self._reset()

            return frame, meta

        # --------------------------------------------------
        # 2) VERIFY — same candidate only
        # --------------------------------------------------
        px, py, pw, ph = self.candidate_box
        p_xyxy = (px, py, px + pw, py + ph)

        best_iou = 0.0
        for b in bats:
            bx, by, bw, bh = b["box"]
            b_xyxy = (bx, by, bx + bw, by + bh)
            i = iou_xyxy(b_xyxy, p_xyxy)
            best_iou = max(best_iou, i)

        meta["best_iou"] = float(best_iou)

        if best_iou >= self.iou_thresh:
            self.consec_count += 1
            self.center_history.append(center_xywh(self.candidate_box))
        else:
            self._reset()
            return frame, meta

        meta["consec_count"] = int(self.consec_count)

        # --------------------------------------------------
        # 3) CONFIRMATION (✅ CLAMPED BOX)
        # --------------------------------------------------
        if self.consec_count >= self.consec_required:
            centers = np.stack(self.center_history[-self.consec_required:])
            mean_center = centers.mean(axis=0)

            h, w = frame.shape[:2]

            x = int(mean_center[0] - pw / 2)
            y = int(mean_center[1] - ph / 2)

            # ✅ CLAMP TO IMAGE BOUNDS
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            pw_c = min(pw, w - x)
            ph_c = min(ph, h - y)

            final_box = (x, y, int(pw_c), int(ph_c))

            self.state = self.CONFIRMED
            self.confirmed_bbox = final_box

            meta["state"] = "CONFIRMED"
            meta["batsman_confirmed"] = True
            meta["batsman_bbox"] = list(map(int, final_box))

        return frame, meta
