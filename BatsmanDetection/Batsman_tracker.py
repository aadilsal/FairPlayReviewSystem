import cv2

class BatsmanTracker:
    """
    Wrapper for single-object tracking used after batsman is confirmed.
    Uses OpenCV CSRT (fallback to KCF).
    Compatible with both old (cv2.TrackerX) and new (cv2.legacy.TrackerX) OpenCV versions.
    """

    def __init__(self):
        self.tracker = None

    def _create_tracker(self):
        # 1. Try CSRT (Best accuracy, slower)
        try:
            # New OpenCV (4.5+)
            return cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            try:
                # Old OpenCV (<4.5)
                return cv2.TrackerCSRT_create()
            except AttributeError:
                pass

        # 2. Try KCF (Faster, less accurate)
        try:
            return cv2.legacy.TrackerKCF_create()
        except AttributeError:
            try:
                return cv2.TrackerKCF_create()
            except AttributeError:
                pass
        
        # 3. Try MOSSE (Fastest, least accurate)
        try:
            return cv2.legacy.TrackerMOSSE_create()
        except AttributeError:
            try:
                return cv2.TrackerMOSSE_create()
            except AttributeError:
                print("[ERROR] Could not create any tracker. Ensure 'opencv-contrib-python' is installed.")
                return None

    def init_tracker(self, frame, bbox):
        """
        Initialize tracker with bbox = (x,y,w,h) on provided frame.
        Returns True on success.
        """
        self.tracker = self._create_tracker()
        if self.tracker is None:
            return False
            
        try:
            return self.tracker.init(frame, tuple(map(int, bbox)))
        except Exception as e:
            print(f"[WARN] Tracker init failed: {e}")
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