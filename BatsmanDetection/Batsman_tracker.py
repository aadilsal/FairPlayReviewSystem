import cv2

class BatsmanTracker:
    """
    Wrapper for single-object tracking used after batsman is confirmed.
    Uses OpenCV CSRT (fallback to KCF/MOSSE).
    """

    def __init__(self):
        self.tracker = None

    def _create_tracker(self):
        try:
            return cv2.TrackerCSRT_create()
        except Exception:
            try:
                return cv2.TrackerKCF_create()
            except Exception:
                return cv2.TrackerMOSSE_create()

    def init_tracker(self, frame, bbox):
        """
        Initialize tracker with bbox = (x,y,w,h) on provided frame.
        Returns True on success.
        """
        self.tracker = self._create_tracker()
        try:
            return self.tracker.init(frame, tuple(map(int, bbox)))
        except Exception:
            self.tracker = None
            return False

    def update(self, frame):
        """
        Update tracker on new frame.
        Returns (ok, bbox) where bbox is (x,y,w,h) if ok True.
        """
        if self.tracker is None:
            return False, None
        ok, bbox = self.tracker.update(frame)
        return ok, bbox