import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Generator, Tuple

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, target_size=(640, 360), frame_step=1):
        self.target_size = target_size
        self.frame_step = frame_step
        logger.info(f"    🎬 VideoProcessor initialized (size={target_size}, step={frame_step})")

    def validate_video(self, video_path: Path) -> Tuple[bool, str]:
        logger.info(f"    🔍 Validating video: {video_path.name}")
        if not video_path.exists():
            logger.error("    ❌ Video file does not exist")
            return False, "file_missing"
        # basic check: able to open
        cap = cv2.VideoCapture(str(video_path))
        ok, _ = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if not ok:
            logger.error("    ❌ Cannot read frames from video")
            return False, "cannot_read_frames"
        
        logger.info(f"    ✓ Video valid: {frame_count} frames @ {fps:.2f} fps")
        return True, "ok"

    def extract_frames(self, video_path: Path) -> Generator[np.ndarray, None, None]:
        cap = cv2.VideoCapture(str(video_path))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.frame_step == 0:
                yield frame
            idx += 1
        cap.release()

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, self.target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype("float32") / 255.0
        return frame
