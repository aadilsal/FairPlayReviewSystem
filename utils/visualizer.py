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
        cv2.rectangle(vis_frame, (bx, by), (bx+bw, by+bh), color, 2)
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

def visualize_pads(det_pads, vis_frame, show_feet=False, use_union=True):
    if not det_pads:
        return
    
    pads = [p for p in det_pads if "Foot" not in p.get("label", "")]
    if not pads:
        return

    if use_union and len(pads) > 1:
        coords = []
        for p in pads:
            x, y, w, h = p["box"]
            coords.append([x, y, x + w, y + h])
        
        coords = np.array(coords)
        
        ux1 = np.min(coords[:, 0])
        uy1 = np.min(coords[:, 1])
        ux2 = np.max(coords[:, 2])
        uy2 = np.max(coords[:, 3])
        
        display_list = [{
            "label": "Total_Pads",
            "box": [ux1, uy1, ux2 - ux1, uy2 - uy1]
        }]
    else:
        display_list = pads
        
    for p in display_list:
        px, py, pw, ph = map(int, p["box"])
        cv2.rectangle(vis_frame, (px, py), (px + pw, py + ph), (255, 255, 0), 2)
        cv2.putText(vis_frame, p["label"], (px, max(0, py - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


def _ball_center(info):
    if not info:
        return None
    pos = info.get('interpolated_position')
    if pos is not None and len(pos) >= 2:
        return (float(pos[0]), float(pos[1]))
    box = info.get('box')
    if box and len(box) >= 4:
        return (float(box[0] + box[2] / 2.0), float(box[1] + box[3] / 2.0))
    return None


def _indexed_ball_points(det_ball, upto_frame_idx):
    pts = []
    if not det_ball:
        return pts
    last = min(int(upto_frame_idx), len(det_ball) - 1)
    for i in range(0, last + 1):
        info = det_ball[i]
        if info is None or info.get('ghost', False):
            continue
        center = _ball_center(info)
        if center is not None:
            pts.append((i, (int(center[0]), int(center[1]))))
    return pts


def visualize_ball(det_ball, vis_frame, frame_idx, draw_path=True):
    if not det_ball or frame_idx >= len(det_ball):
        return

    color_map = {
        'yolo': (0, 255, 0),
        'yolo-rescue': (0, 255, 255),
        'csrt-agreed': (255, 255, 0),
        'kinematic': (0, 0, 255),
        'edge-suspected': (0, 0, 255),
    }

    if draw_path:
        path_points = [p for _fi, p in _indexed_ball_points(det_ball, frame_idx)]
        if len(path_points) > 1:
            arr = np.array(path_points, dtype=np.int32)
            cv2.polylines(vis_frame, [arr], False, (255, 220, 100), 2, cv2.LINE_AA)

    current_info = det_ball[frame_idx]
    if current_info is not None and not current_info.get('ghost', False):
        source = current_info.get('source', 'yolo')
        color = color_map.get(source, (0, 255, 0))
        raw_box = current_info.get('box', [0, 0, 0, 0])
        center = _ball_center(current_info)
        if center is None:
            return
        curr_center = (int(center[0]), int(center[1]))
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

def _draw_dashed_polyline(img, pts, color, thickness=2, dash_length=10, gap_length=6):
    if len(pts) < 2:
        return
    for i in range(len(pts) - 1):
        p1 = np.array(pts[i], dtype=np.float32)
        p2 = np.array(pts[i + 1], dtype=np.float32)
        seg_len = float(np.linalg.norm(p2 - p1)) + 1e-6
        u = (p2 - p1) / seg_len
        dist = 0.0
        draw = True
        while dist < seg_len:
            step = min(dash_length if draw else gap_length, seg_len - dist)
            a = p1 + u * dist
            dist += step
            if draw:
                b = p1 + u * min(dist, seg_len)
                cv2.line(
                    img,
                    (int(round(a[0])), int(round(a[1]))),
                    (int(round(b[0])), int(round(b[1]))),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            draw = not draw


def _as_polyline_points(data):
    pts = []
    for p in data or []:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        pts.append((int(round(float(p[0]))), int(round(float(p[1])))))
    return pts


def visualize_lbw_overlay(vis_frame, lbw_overlay, frame_idx, det_ball=None):
    if not lbw_overlay:
        return

    wl = lbw_overlay.get("wicket_line")
    if wl and len(wl) == 2:
        p0 = tuple(map(int, wl[0]))
        p1 = tuple(map(int, wl[1]))
        cv2.line(vis_frame, p0, p1, (200, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(
            vis_frame,
            "Wicket line",
            (p1[0] + 6, p1[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 255),
            1,
            cv2.LINE_AA,
        )

    impact_frame_idx = lbw_overlay.get("impact_frame_idx")
    pitch_frame_idx = lbw_overlay.get("pitch_frame_idx")
    if pitch_frame_idx is None:
        pitch_frame_idx = lbw_overlay.get("bounce_frame")

    # Trajectory policy:
    # 1) Draw detected-ball path until impact.
    # 2) Change color once pitch is reached.
    # 3) After impact: only pad-contact gets projected path; bat-contact stops path.
    draw_until = frame_idx
    if impact_frame_idx is not None and frame_idx >= int(impact_frame_idx):
        draw_until = min(frame_idx, int(impact_frame_idx))

    indexed_points = _indexed_ball_points(det_ball, draw_until)
    if indexed_points:
        if pitch_frame_idx is None:
            all_pts = [p for _fi, p in indexed_points]
            if len(all_pts) > 1:
                arr = np.array(all_pts, dtype=np.int32)
                cv2.polylines(vis_frame, [arr], False, (255, 220, 100), 2, cv2.LINE_AA)
        else:
            pfi = int(pitch_frame_idx)
            pre_pts = [p for fi, p in indexed_points if fi <= pfi]
            post_pts = [p for fi, p in indexed_points if fi >= pfi]
            if len(pre_pts) > 1:
                arr = np.array(pre_pts, dtype=np.int32)
                cv2.polylines(vis_frame, [arr], False, (255, 220, 100), 2, cv2.LINE_AA)
            if len(post_pts) > 1:
                arr = np.array(post_pts, dtype=np.int32)
                cv2.polylines(vis_frame, [arr], False, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        # Fallback for legacy overlays without det_ball context.
        fitted = lbw_overlay.get("fitted_polyline") or []
        fitted_start = int(lbw_overlay.get("fitted_start_frame", 0) or 0)
        total_visible = max(0, min(len(fitted), frame_idx - fitted_start + 1))
        fitted_visible = _as_polyline_points(fitted[:total_visible])
        if len(fitted_visible) > 1:
            arr = np.array(fitted_visible, dtype=np.int32)
            cv2.polylines(vis_frame, [arr], False, (255, 220, 100), 2, cv2.LINE_AA)

    ext = lbw_overlay.get("projected_from_impact_polyline") or lbw_overlay.get("predicted_extension") or []
    has_pad_contact = bool(lbw_overlay.get("pad_contact", False))
    impact_target = str(lbw_overlay.get("impact_target") or "").lower()
    should_project = has_pad_contact or impact_target == "pad"
    if len(ext) > 1 and should_project and impact_frame_idx is not None and frame_idx >= int(impact_frame_idx):
        # Reveal extension gradually after impact so it appears animated in sequence.
        ext_visible_count = min(len(ext), max(0, (int(frame_idx) - int(impact_frame_idx) + 1) * 2))
        ext_visible = _as_polyline_points(ext[:ext_visible_count])
        if len(ext_visible) > 1:
            _draw_dashed_polyline(vis_frame, ext_visible, (255, 90, 210), thickness=2)

    pp = lbw_overlay.get("pitch_point")
    if pp is not None and pitch_frame_idx is not None and frame_idx >= int(pitch_frame_idx):
        c = (int(pp[0]), int(pp[1]))
        cv2.circle(vis_frame, c, 8, (0, 255, 255), 2)
        cv2.putText(
            vis_frame,
            "Pitch",
            (c[0] + 10, c[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    ip = lbw_overlay.get("impact_point")
    if ip is not None and impact_frame_idx is not None and frame_idx >= int(impact_frame_idx):
        c = (int(ip[0]), int(ip[1]))
        cv2.circle(vis_frame, c, 8, (0, 165, 255), 2)
        cv2.putText(
            vis_frame,
            "Impact",
            (c[0] + 10, c[1] + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    sp = lbw_overlay.get("stump_intersection")
    if sp is not None and impact_frame_idx is not None and frame_idx >= int(impact_frame_idx):
        c = (int(sp[0]), int(sp[1]))
        cv2.circle(vis_frame, c, 10, (100, 255, 100), 2)
        cv2.putText(
            vis_frame,
            "Stumps",
            (c[0] + 12, c[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 255, 100),
            2,
            cv2.LINE_AA,
        )

    y0 = 52
    lines = [
        f"Pitch inline: {'Y' if lbw_overlay.get('pitch_inline') else 'N'}",
        f"Impact inline: {'Y' if lbw_overlay.get('impact_inline') else 'N'}",
        f"Wickets: {'Hitting' if lbw_overlay.get('wickets_hitting') else 'Missing'}",
        f"Decision: {lbw_overlay.get('decision', 'N/A')}",
    ]
    rsn = lbw_overlay.get("reason")
    if rsn:
        lines.append(str(rsn))
    for i, line in enumerate(lines):
        y = y0 + i * 22
        cv2.putText(
            vis_frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def visualize_frame(
    frame,
    det_ball,
    det_persons,
    det_batsman_box,
    det_wickets,
    det_bats,
    det_pads,
    det_pose,
    frame_idx,
    lbw_overlay=None,
):
    vis_frame = frame

    # Keep live pass clean: trajectory is drawn only by LBW overlay after analysis.
    visualize_ball(det_ball, vis_frame, frame_idx, draw_path=False)
    visualize_bat(det_bats, vis_frame)
    visualize_batsman(det_batsman_box, det_persons, vis_frame)
    visualize_wicket(det_wickets, vis_frame)
    visualize_pose(det_pose, vis_frame)
    visualize_pads(det_pads, vis_frame)

    visualize_lbw_overlay(vis_frame, lbw_overlay, frame_idx, det_ball=det_ball)

    _draw_frame_info(vis_frame, frame_idx)

    return vis_frame