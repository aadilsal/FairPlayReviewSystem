import cv2
import numpy as np

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
    text = f"Frame: {frame_idx}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 0, 255), 2, cv2.LINE_AA)

def draw_pitch_overlay(frame, far_box, near_box):
    if far_box is None or near_box is None:
        return

    fx, fy, fw, fh = map(int, far_box)
    nx, ny, nw, nh = map(int, near_box)

    pts = np.array([
        [fx, fy + fh],          # Far Bottom-Left
        [fx + fw, fy + fh],     # Far Bottom-Right
        [nx + nw, ny + nh],     # Near Bottom-Right
        [nx, ny + nh]           # Near Bottom-Left
    ], np.int32).reshape((-1, 1, 2))

    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 255)) 
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.polylines(frame, [pts], True, (0, 200, 200), 2)

def visualize_wicket(det_wickets, vis_frame):
    if not det_wickets:
        return

    far_wkt = None
    near_wkt = None

    for w in det_wickets:
        box = w["box"]
        lbl = w.get("label", "Wicket")
        
        if "Far" in lbl: far_wkt = box
        elif "Near" in lbl: near_wkt = box

        wx, wy, ww, wh = map(int, box)
        color = (0, 140, 255) if "Far" in lbl else (0, 255, 0)
        cv2.rectangle(vis_frame, (wx, wy), (wx + ww, wy + wh), color, 2)
        cv2.putText(vis_frame, lbl, (wx, wy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    draw_pitch_overlay(vis_frame, far_wkt, near_wkt)


def visualize_batsman(det_batsman_box, det_persons, vis_frame):
    if det_batsman_box:
        bx, by, bw, bh = det_batsman_box
        color = (255, 0, 0) 
        cv2.rectangle(vis_frame, (bx, by), (bx+bw, by+bh), color, 3)
        cv2.putText(vis_frame, "Batsman", (bx, max(0, by-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        for p in det_persons:
            px, py, pw, ph, _ = p
            px, py, pw, ph = map(int, [px, py, pw, ph])
            cv2.rectangle(vis_frame, (px, py), (px+pw, py+ph), (0, 255, 0), 2)
            cv2.putText(vis_frame, "Person", (px, max(0, py - 5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def visualize_bat(det_bats, vis_frame):
    if not det_bats:
        return
    for b in det_bats:
        if "box" in b:
            bx, by, bw, bh = map(int, b["box"])
            color = (255, 0, 255) 
            cv2.rectangle(vis_frame, (bx, by), (bx+bw, by+bh), color, 2)
            cv2.putText(vis_frame, "Bat", (bx, max(0, by - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def visualize_pads(det_pads, vis_frame):
    if not det_pads:
        return
    for p in det_pads:
        if "box" in p:
            px, py, pw, ph = map(int, p["box"])
            color = (255, 255, 0)
            cv2.rectangle(vis_frame, (px, py), (px+pw, py+ph), color, 2)
            cv2.putText(vis_frame, "Pad", (px, max(0, py - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def visualize_ball(det_ball, vis_frame, frame_idx):
    if not det_ball or frame_idx >= len(det_ball):
        return
        
    color_map = {
        'yolo': (0, 255, 0),         # Green
        'yolo-rescue': (0, 255, 255), # Yellow
        'csrt-agreed': (255, 255, 0), # Cyan
        'kinematic': (0, 0, 255)      # Red
    }

    prev_pos = None

    for i in range(frame_idx + 1):
        info = det_ball[i]
        if info is None or info.get('ghost', False):
            continue

        if 'interpolated_position' in info and info['interpolated_position'] is not None:
            pos = info['interpolated_position']
        else:
            box = info.get('box', [0, 0, 0, 0])
            pos = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
        
        pos_int = (int(pos[0]), int(pos[1]))
        source = info.get('source', 'yolo')
        color = color_map.get(source, (0, 255, 0))

        if prev_pos is not None:
            cv2.line(vis_frame, prev_pos, pos_int, color, 2)
        
        cv2.circle(vis_frame, pos_int, 2, color, -1)
        prev_pos = pos_int

        if i == frame_idx:
            raw_box = info.get('box', [0, 0, 0, 0])
            curr_center = (int(raw_box[0] + raw_box[2]/2), int(raw_box[1] + raw_box[3]/2))
            
            radius = int((raw_box[2] + raw_box[3]) / 4) if raw_box[2] > 0 else 5
            
            cv2.circle(vis_frame, curr_center, radius, color, 2) 
            cv2.putText(vis_frame, "Ball", (curr_center[0] + radius + 5, curr_center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def visualize_pose(det_pose, vis_frame):
    if not det_pose:
        return

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

def visualize_frame(frame, det_ball, det_persons, det_batsman_box, det_wickets, det_bats, det_pads, det_pose, frame_idx):
    vis_frame = frame

    visualize_ball(det_ball, vis_frame, frame_idx) 
    visualize_bat(det_bats, vis_frame)
    visualize_batsman(det_batsman_box, det_persons, vis_frame)
    visualize_wicket(det_wickets, vis_frame)
    visualize_pose(det_pose, vis_frame)
    visualize_pads(det_pads, vis_frame)

    _draw_frame_info(vis_frame, frame_idx)

    return vis_frame