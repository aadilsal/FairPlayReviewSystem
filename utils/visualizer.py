# file: visualizer.py
import cv2
import numpy as np

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
SKELETON_PAIRS = [
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
    (5, 6),              # shoulders
    (11, 12),            # hips
    (5, 11), (6, 12)     # torso
]

def _draw_frame_info(frame, frame_idx):
    """Draws only the frame number in the top-left corner"""
    text = f"Frame: {frame_idx}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 0, 255), 2, cv2.LINE_AA)

def _draw_pitch_overlay(frame, far_box, near_box):
    """Draws pitch area"""
    if not far_box or not near_box:
        return

    fx, fy, fw, fh = map(int, far_box)
    nx, ny, nw, nh = map(int, near_box)

    pts = np.array([
        [fx, fy + fh],          # Far Bottom-Left
        [fx + fw, fy + fh],     # Far Bottom-Right
        [nx + nw, ny + nh],     # Near Bottom-Right
        [nx, ny + nh]           # Near Bottom-Left
    ], np.int32)

    pts = pts.reshape((-1, 1, 2))
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 255))
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts], True, (0, 200, 200), 2)

def visualize_frame(frame, det_ball, det_persons, det_batsman_box, det_wickets, det_bats, det_pads, det_pose, frame_idx):
    """
    Draws detections + Frame Number. 
    """
    vis_frame = frame

    # 0. Draw Pitch Overlay
    far_wkt = None
    near_wkt = None
    if det_wickets:
        for w in det_wickets:
            label = w.get("label", "")
            if "Far" in label:
                far_wkt = w["box"]
            elif "Near" in label:
                near_wkt = w["box"]
    if far_wkt and near_wkt:
        _draw_pitch_overlay(vis_frame, far_wkt, near_wkt)


    # 1. Draw Batsman (Blue) + Label
    if det_batsman_box:
        bx, by, bw, bh = det_batsman_box
        color = (255, 0, 0) 
        cv2.rectangle(vis_frame, (bx, by), (bx+bw, by+bh), color, 3)
        cv2.putText(vis_frame, "Batsman", (bx, max(0, by-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        # 2. Draw ALL Persons (Green) + Label
        for p in det_persons:
            px, py, pw, ph, _ = p
            px, py, pw, ph = map(int, [px, py, pw, ph])
            cv2.rectangle(vis_frame, (px, py), (px+pw, py+ph), (0, 255, 0), 2)
            cv2.putText(vis_frame, "Person", (px, max(0, py - 5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 3. Draw Bats (Magenta) + Label
    if det_bats:
        for b in det_bats:
            if "box" in b:
                bx, by, bw, bh = map(int, b["box"])
                color = (255, 0, 255) # Magenta
                cv2.rectangle(vis_frame, (bx, by), (bx+bw, by+bh), color, 2)
                cv2.putText(vis_frame, "Bat", (bx, max(0, by - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 4. Draw Pads (Cyan) + Label
    if det_pads:
        for p in det_pads:
            if "box" in p:
                px, py, pw, ph = map(int, p["box"])
                color = (255, 255, 0) # Cyan (Yellow/Green mix)
                cv2.rectangle(vis_frame, (px, py), (px+pw, py+ph), color, 2)
                cv2.putText(vis_frame, "Pad", (px, max(0, py - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # # 5. Draw Ball (Red) + Label
    # if det_ball:
    #     bx, by, br = 0, 0, 0
    #     valid_ball = False
        
    #     # Handle Dictionary Format {'box': [x,y,w,h], 'conf': ...}
    #     if isinstance(det_ball, dict) and "box" in det_ball:
    #         x, y, w, h = det_ball["box"]
    #         bx = int(x + w // 2)
    #         by = int(y + h // 2)
    #         br = int(max(w, h) // 2)
    #         valid_ball = True
            
    #     # Handle legacy tuple format (x, y, radius) just in case
    #     elif isinstance(det_ball, (list, tuple)) and len(det_ball) >= 3:
    #         bx, by, br = int(det_ball[0]), int(det_ball[1]), int(det_ball[2])
    #         valid_ball = True

    #     if valid_ball:
    #         cv2.circle(vis_frame, (bx, by), br, (0, 0, 255), 2)
    #         cv2.circle(vis_frame, (bx, by), 2, (0, 0, 255), -1)
    #         cv2.putText(vis_frame, "Ball", (bx + br + 5, by), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# 5. Draw Ball (Red) + Label
    if det_ball:
        bx, by, br = 0, 0, 0
        valid_ball = False
        
        # --- Draw Crop Area (Blue Box) ---
        if isinstance(det_ball, dict) and "crop_box" in det_ball:
            cx1, cy1, cx2, cy2 = map(int, det_ball["crop_box"])
            cv2.rectangle(vis_frame, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)
            cv2.putText(vis_frame, "Crop Area", (cx1 + 5, cy1 + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # --- Draw ROI Search Area (Green Box) ---
        if isinstance(det_ball, dict) and "roi_box" in det_ball:
            # roi_box is [x1, y1, x2, y2]
            rx1, ry1, rx2, ry2 = det_ball["roi_box"]
            
            # Draw rectangle (Green, thin)
            cv2.rectangle(vis_frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 1)
            
            # Label it "ROI" (Small text)
            cv2.putText(vis_frame, "ROI", (rx1, ry1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Handle Dictionary Format {'box': [x,y,w,h], 'conf': ...}
        if isinstance(det_ball, dict) and "box" in det_ball:
            # Note: Ensure your detector returns box as [x_center, y_center, w, h] or [x1, y1, w, h]
            # Assuming [x_center, y_center, w, h] based on your circle logic below:
            x, y, w, h = det_ball["box"]
            bx = int(x) # if x is center
            by = int(y) # if y is center
            # OR if x,y are top-left:
            # bx = int(x + w // 2)
            # by = int(y + h // 2)
            
            # Let's stick to the logic you had (assuming x,y are top-left based on w//2 add):
            bx = int(x + w // 2) 
            by = int(y + h // 2)
            br = int(max(w, h) // 2)
            
            # Draw GHOST indicator if applicable
            if det_ball.get('ghost', False):
                # Draw Dashed/Different color for Ghost (e.g., Cyan)
                cv2.circle(vis_frame, (bx, by), br, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(vis_frame, "Ghost", (bx + br + 5, by), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                # Standard Detection (Red)
                cv2.circle(vis_frame, (bx, by), br, (0, 0, 255), 2)
                cv2.circle(vis_frame, (bx, by), 2, (0, 0, 255), -1)
                cv2.putText(vis_frame, "Ball", (bx + br + 5, by), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    # 6. Draw Wickets (Orange/Green) + Label
    if det_wickets:
        for w in det_wickets:
            wx, wy, ww, wh = map(int, w["box"])
            lbl = w.get("label", "Wicket")
            color = (0, 140, 255) if "Far" in lbl else (0, 255, 0)
            cv2.rectangle(vis_frame, (wx, wy), (wx+ww, wy+wh), color, 2)
            cv2.putText(vis_frame, lbl, (wx, wy-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 7. Draw Pose
    for person_kps in det_pose:
        for (i, j) in SKELETON_PAIRS:
            if i < len(person_kps) and j < len(person_kps):
                pt1 = (int(person_kps[i][0]), int(person_kps[i][1]))
                pt2 = (int(person_kps[j][0]), int(person_kps[j][1]))
                if pt1[0] > 0 and pt2[0] > 0:
                    cv2.line(vis_frame, pt1, pt2, (0, 255, 0), 2)
        for kp in person_kps:
            kx, ky = int(kp[0]), int(kp[1])
            if kx > 0 and ky > 0:
                cv2.circle(vis_frame, (kx, ky), 3, (0, 255, 255), -1)

    # 8. Draw Frame Number
    _draw_frame_info(vis_frame, frame_idx)

    return vis_frame