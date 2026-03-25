from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from API.core.supabase_client import (
    MATCHES_TABLE,
    WICKET_CONFIGURATIONS_TABLE,
    supabase_client,
)
from API.utils.file_handler import delete_file


def _pick_near_far(detections: List[Dict[str, Any]]) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    near_box = None
    far_box = None
    for d in detections or []:
        label = str(d.get("label") or "")
        box = d.get("box")
        if not box or not isinstance(box, list) or len(box) != 4:
            continue
        if "Near" in label and near_box is None:
            near_box = [float(x) for x in box]
        if "Far" in label and far_box is None:
            far_box = [float(x) for x in box]
    return near_box, far_box


def _read_image(image_path: str) -> Any:
    payload = Path(image_path).read_bytes()
    if not payload:
        raise RuntimeError("Empty image payload")
    arr = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Unable to decode image")
    return frame


def _bootstrap_wicket_detector_imports() -> None:
    # Mirror DetectionService bootstrap so this can be used from API module.
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    required = [
        root,
        os.path.join(root, "WicketDetection"),
        os.path.join(root, "utils"),
    ]
    import sys

    for p in required:
        if p not in sys.path:
            sys.path.append(p)


class WicketConfigService:
    @staticmethod
    def _upsert_row(payload: Dict[str, Any]) -> Dict[str, Any]:
        # Single round-trip write (requires unique constraint on match_id,user_id).
        resp = (
            supabase_client.table(WICKET_CONFIGURATIONS_TABLE)
            .upsert(payload, on_conflict="match_id,user_id")
            .execute()
        )
        if not resp.data:
            raise HTTPException(status_code=500, detail="Failed to persist wicket configuration")
        return resp.data[0]

    @staticmethod
    def _ensure_match_owned(match_id: int, user_id: int) -> Dict[str, Any]:
        resp = supabase_client.table(MATCHES_TABLE).select("id,user_id").eq("id", match_id).execute()
        if not resp.data:
            raise HTTPException(status_code=404, detail="Match not found")
        row = resp.data[0]
        # Migration-safe: if user_id is missing/null in DB, allow access (older schema).
        if row.get("user_id") is not None and int(row.get("user_id") or 0) != int(user_id):
            raise HTTPException(status_code=403, detail="Not allowed for this match")
        return row

    @staticmethod
    def get_config(match_id: int, user_id: int) -> Optional[Dict[str, Any]]:
        WicketConfigService._ensure_match_owned(match_id, user_id)
        resp = (
            supabase_client.table(WICKET_CONFIGURATIONS_TABLE)
            .select("*")
            .eq("match_id", match_id)
            .eq("user_id", user_id)
            .execute()
        )
        return resp.data[0] if resp.data else None

    @staticmethod
    def upsert_manual_config(
        match_id: int,
        user_id: int,
        near_box: Optional[List[float]],
        far_box: Optional[List[float]],
        *,
        configured: bool = True,
    ) -> Dict[str, Any]:
        WicketConfigService._ensure_match_owned(match_id, user_id)
        payload = {
            "match_id": match_id,
            "user_id": user_id,
            "configured": bool(configured),
            "near_box": near_box,
            "far_box": far_box,
            "status": "completed",
            "error_message": None,
        }

        return WicketConfigService._upsert_row(payload)

    @staticmethod
    def mark_processing(match_id: int, user_id: int) -> Dict[str, Any]:
        WicketConfigService._ensure_match_owned(match_id, user_id)
        payload = {
            "match_id": match_id,
            "user_id": user_id,
            "status": "processing",
            "error_message": None,
        }
        return WicketConfigService._upsert_row(payload)

    @staticmethod
    def auto_configure_from_image_path(
        match_id: int,
        user_id: int,
        image_path: str,
        wicket_conf: float = 0.25,
        display: bool = False,
    ) -> Dict[str, Any]:
        WicketConfigService._ensure_match_owned(match_id, user_id)

        # Lightweight service-level logging (route logs request context)
        try:
            import logging

            logger = logging.getLogger("fairplay.api.wicket_config")
            logger.info(
                "Auto-configure wicket from image path match_id=%s user_id=%s path=%s",
                match_id,
                user_id,
                image_path,
            )
        except Exception:
            pass

        try:
            _bootstrap_wicket_detector_imports()
            try:
                from wicket_detector import detect_wicket  # type: ignore
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Wicket detector not available: {exc}") from exc

            frame = _read_image(image_path)
            rendered_frame, dets = detect_wicket(frame, conf=wicket_conf)
            logger.info(
                "Wicket detections match_id=%s user_id=%s count=%s dets=%s",
                match_id,
                user_id,
                len(dets or []),
                dets or [],
            )
            near_box, far_box = _pick_near_far(dets)

            annotated_local_path = None
            if rendered_frame is not None:
                try:
                    out_dir = Path("outputs") / "wicket-config"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    annotated_local_path = str(out_dir / f"annotated_{match_id}_{user_id}_{uuid.uuid4().hex}.jpg")
                    cv2.imwrite(annotated_local_path, rendered_frame)
                    logger.info("Saved annotated wicket image: %s", annotated_local_path)
                except Exception as exc:
                    logger.warning("Failed saving annotated wicket image: %s", exc)

            if display and rendered_frame is not None:
                try:
                    window_name = "Wicket Detection Preview"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(window_name, rendered_frame)
                    logger.info(
                        "Displaying wicket detection preview window match_id=%s user_id=%s",
                        match_id,
                        user_id,
                    )
                    # Keep window responsive for up to 8s, allow close by q/ESC.
                    started = time.time()
                    while (time.time() - started) < 8.0:
                        key = cv2.waitKey(50) & 0xFF
                        if key in (27, ord("q")):
                            break
                    cv2.destroyWindow(window_name)
                except Exception as exc:
                    logger.warning("Could not display wicket detection preview: %s", exc)

            if not near_box and not far_box:
                raise HTTPException(status_code=422, detail="No wicket detected in provided image")

            # If only one side is detected, still persist and allow manual edit later.
            record = WicketConfigService.upsert_manual_config(
                match_id=match_id,
                user_id=user_id,
                near_box=near_box,
                far_box=far_box,
                configured=bool(near_box and far_box),
            )

            # Attach annotated reference (best-effort)
            try:
                patch: Dict[str, Any] = {"match_id": match_id, "user_id": user_id}
                if annotated_local_path:
                    patch["annotated_image_path"] = annotated_local_path
                if len(patch.keys()) > 2:
                    WicketConfigService._upsert_row(patch)
            except Exception as exc:
                logger.warning("Failed to persist annotated/source image refs: %s", exc)

            record["raw_detections"] = dets or []
            return record
        finally:
            delete_file(image_path)

    @staticmethod
    def mark_failed(match_id: int, user_id: int, error_message: str) -> None:
        payload = {
            "match_id": match_id,
            "user_id": user_id,
            "status": "failed",
            "error_message": error_message,
            "configured": False,
        }
        WicketConfigService._upsert_row(payload)

