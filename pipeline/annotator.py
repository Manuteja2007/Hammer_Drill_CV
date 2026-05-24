from typing import List, Optional
import cv2
from .detector import Detection
from .tracker import KalmanTracker

class Annotator:
    """Pure functions for drawing bounding boxes, trails, and encoding JPEG frames."""
    
    @staticmethod
    def draw_detections(frame, detections: List[Detection]) -> None:
        """Draw bounding boxes and labels for all detections."""
        for d in detections:
            x1, y1, x2, y2 = d.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{d.label} {d.confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    @staticmethod
    def draw_tracker(frame, tracker: KalmanTracker, predicted_pos: Optional[tuple] = None) -> None:
        """Draw the Kalman filter prediction dot and the historical trail."""
        # 1. Prediction dot
        if tracker.initialised and predicted_pos is not None:
            px, py = predicted_pos
            cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)
            
        # 2. Historical trail
        trail = list(tracker.trail)
        for i in range(1, len(trail)):
            if trail[i-1] and trail[i]:
                cv2.line(frame, trail[i-1], trail[i], (255, 0, 0), 2)

    @staticmethod
    def encode_jpeg(frame) -> Optional[bytes]:
        """Convert an OpenCV BGR frame to a JPEG byte string."""
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return buf.tobytes()

    @staticmethod
    def format_multipart_jpeg(data: bytes) -> bytes:
        """Wrap JPEG bytes in MJPEG multipart format."""
        return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
