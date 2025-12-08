import cv2

class BatsmanTracker:
    """
    Wrapper for single-object tracking used after batsman is confirmed.
    Uses OpenCV CSRT (fallback to KCF/MOSSE).
    """

    def __init__(self):
        self.tracker = None

    def _create_tracker(self):
        # Try multiple possible tracker constructors to support different OpenCV builds
        constructors = [
            ("legacy", "TrackerCSRT_create"),
            (None, "TrackerCSRT_create"),
            ("legacy", "TrackerKCF_create"),
            (None, "TrackerKCF_create"),
            ("legacy", "TrackerMOSSE_create"),
            (None, "TrackerMOSSE_create"),
        ]
        for ns, name in constructors:
            try:
                if ns is None:
                    ctor = getattr(cv2, name)
                else:
                    ns_obj = getattr(cv2, ns, None)
                    if ns_obj is None:
                        continue
                    ctor = getattr(ns_obj, name)
                return ctor()
            except Exception:
                continue

        # Older OpenCV had a factory function Tracker_create(type)
        try:
            if hasattr(cv2, 'Tracker_create'):
                return cv2.Tracker_create('KCF')
        except Exception:
            pass

        # As a last resort, return None (caller must handle)
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
            # Some tracker implementations expect init(frame, bbox)
            return self.tracker.init(frame, tuple(map(int, bbox)))
        except Exception:
            # Some older variants require creating via cv2.Tracker_* and calling init similarly
            try:
                # attempt to call under legacy namespace if available
                if hasattr(self.tracker, 'init'):
                    return self.tracker.init(frame, tuple(map(int, bbox)))
            except Exception:
                pass
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