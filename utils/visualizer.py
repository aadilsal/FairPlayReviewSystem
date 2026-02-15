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

def _draw_frame_info(frame, frame_idx, pitch_status=None):
    """Draws the frame number and optional pitch status in the top-left corner"""
    text = f"Frame: {frame_idx}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2, cv2.LINE_AA)
    if pitch_status:
        status_text = f"Pitch: {pitch_status}"
        cv2.putText(frame, status_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)

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


def _draw_pitch_model_overlay(frame, pitch_model, fill=True):
    if not pitch_model:
        return
    left_line = pitch_model.get("left_line")
    right_line = pitch_model.get("right_line")
    y_top = pitch_model.get("top_y")
    y_bottom = pitch_model.get("bottom_y")
    polygon = pitch_model.get("polygon")
    if left_line is None or right_line is None or y_top is None or y_bottom is None:
        return

    a_left, b_left = left_line
    a_right, b_right = right_line
    h, w = frame.shape[:2]
    y_top = int(max(0, min(h - 1, y_top)))
    y_bottom = int(max(0, min(h - 1, y_bottom)))
    if y_bottom <= y_top:
        return

    x_left_top = int(max(0, min(w - 1, a_left * y_top + b_left)))
    x_left_bottom = int(max(0, min(w - 1, a_left * y_bottom + b_left)))
    x_right_top = int(max(0, min(w - 1, a_right * y_top + b_right)))
    x_right_bottom = int(max(0, min(w - 1, a_right * y_bottom + b_right)))

    if polygon:
        pts = np.array(polygon, np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        pts = pts.astype(np.int32).reshape((-1, 1, 2))
    else:
        pts = np.array([
            [x_left_top, y_top],
            [x_right_top, y_top],
            [x_right_bottom, y_bottom],
            [x_left_bottom, y_bottom]
        ], np.int32).reshape((-1, 1, 2))

    overlay = frame.copy()
    if fill:
        cv2.fillPoly(overlay, [pts], (255, 255, 0))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    color = (255, 255, 0)
    thickness = 4
    cv2.polylines(frame, [pts], True, color, thickness)
    cv2.line(frame, (x_left_top, y_top), (x_right_top, y_top), color, thickness)
    cv2.line(frame, (x_left_bottom, y_bottom), (x_right_bottom, y_bottom), color, thickness)
    cv2.line(frame, (x_left_top, y_top), (x_left_bottom, y_bottom), color, thickness)
    cv2.line(frame, (x_right_top, y_top), (x_right_bottom, y_bottom), color, thickness)
    conf = pitch_model.get("confidence", None)
    if conf is not None:
        cv2.putText(
            frame,
            f"Pitch OK ({conf:.2f})",
            (max(10, x_left_top), max(20, y_top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

def visualize_frame(frame, det_ball, det_persons, det_batsman_box, det_wickets, det_bats, det_pads, det_pose, frame_idx, pitch_model=None, pitch_status=None, wicket_line=None, lbw_decision=None, lbw_status=None):
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

    _draw_pitch_model_overlay(vis_frame, pitch_model, fill=True)
    if det_ball and isinstance(det_ball, dict):
        trajectory = det_ball.get("trajectory")
        if trajectory:
            pts = np.array(trajectory, dtype=np.int32).reshape((-1, 1, 2))
            if len(pts) > 1:
                cv2.polylines(vis_frame, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)

        state = det_ball.get("state")
        if state:
            bounce_point = state.get("bounce_point")
            impact_point = state.get("impact_point")
            hit_point = state.get("hit_point")
            post_path = state.get("post_impact_path")

            if bounce_point:
                bx, by = int(bounce_point[0]), int(bounce_point[1])
                cv2.circle(vis_frame, (bx, by), 6, (0, 255, 255), 2)
                cv2.putText(vis_frame, "Bounce", (bx + 6, by - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if impact_point:
                ix, iy = int(impact_point[0]), int(impact_point[1])
                cv2.circle(vis_frame, (ix, iy), 6, (0, 0, 255), 2)
                cv2.putText(vis_frame, "Impact", (ix + 6, iy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if post_path:
                pts = np.array(post_path, dtype=np.int32).reshape((-1, 1, 2))
                if len(pts) > 1:
                    cv2.polylines(vis_frame, [pts], False, (255, 255, 0), 2, cv2.LINE_AA)
            if hit_point:
                hx, hy = int(hit_point[0]), int(hit_point[1])
                cv2.circle(vis_frame, (hx, hy), 6, (0, 255, 0), 2)
                cv2.putText(vis_frame, "Stumps", (hx + 6, hy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if wicket_line:
        x = int(wicket_line.get("x", 0))
        y_top = int(wicket_line.get("y_top", 0))
        y_bottom = int(wicket_line.get("y_bottom", 0))
        h, w = vis_frame.shape[:2]
        x = max(0, min(w - 1, x))
        y_top = max(0, min(h - 1, y_top))
        y_bottom = max(0, min(h - 1, y_bottom))
        if y_bottom > y_top:
            cv2.line(vis_frame, (x, y_top), (x, y_bottom), (0, 255, 255), 4)
            cv2.putText(vis_frame, "Wicket Line", (max(10, x - 40), max(20, y_top - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if lbw_status or lbw_decision:
        status_text = f"LBW: {lbw_decision or 'NO DECISION'}"
        if lbw_status:
            status_text = f"LBW: {lbw_status}"
        cv2.putText(vis_frame, status_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)


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

    # 5. Draw Ball (Red) + Label
    if det_ball:
        bx, by, br = 0, 0, 0
        valid_ball = False
        
        # Handle Dictionary Format {'box': [x,y,w,h], 'conf': ...}
        if isinstance(det_ball, dict) and "box" in det_ball:
            if det_ball["box"] is not None:
                x, y, w, h = det_ball["box"]
                bx = int(x + w // 2)
                by = int(y + h // 2)
                br = int(max(w, h) // 2)
                valid_ball = True
            
        # Handle legacy tuple format (x, y, radius) just in case
        elif isinstance(det_ball, (list, tuple)) and len(det_ball) >= 3:
             bx, by, br = int(det_ball[0]), int(det_ball[1]), int(det_ball[2])
             valid_ball = True

        if valid_ball:
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
    _draw_frame_info(vis_frame, frame_idx, pitch_status=pitch_status)

    return vis_frame