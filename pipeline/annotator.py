from typing import List, Optional
import cv2
import numpy as np
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
    def draw_hit_miss_overlay(
        frame,
        last_event_type: Optional[str],
        flash_frames_remaining: int,
        current_streak: int,
        total_hits: int,
        total_misses: int,
    ) -> None:
        """
        Draw hit/miss visual feedback on the frame.

        - Persistent HUD in top-left: hit count, miss count, streak
        - Compact event label without obscuring the ball with a full-frame flash
        """
        height, width = frame.shape[:2]
        panel_width = min(246, max(1, width - 16))
        panel_height = min(68, max(1, height - 16))
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), (35, 48, 53), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, f"Taps {total_hits}  |  Streak {current_streak}", (16, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 245, 240), 1)
        label = f"Streak breaks {total_misses}"
        color = (195, 210, 210)
        if flash_frames_remaining > 0:
            if last_event_type == "HIT":
                label, color = "Tap counted", (135, 220, 175)
            elif last_event_type == "REJECTED":
                label, color = "Tap ignored - review hand check", (100, 195, 245)
            elif last_event_type == "MISS":
                label, color = "Streak break (inferred)", (155, 200, 230)
        font_scale = min(0.45, (panel_width - 16) / max(1, cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0]))
        cv2.putText(frame, label, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

    @staticmethod
    def draw_hand_check(frame, decision):
        from .hand_contact import FINGER_CHAINS

        color = (0, 180, 255) if not decision.allowed else (255, 220, 0)
        for hand in decision.hands:
            points = hand.landmarks.astype(int)
            for chain in FINGER_CHAINS:
                for start, end in zip(chain, chain[1:]):
                    cv2.line(frame, tuple(points[start]), tuple(points[end]), color, 2)
            for point in points:
                cv2.circle(frame, tuple(point), 2, color, -1)
            minimum = points.min(axis=0)
            maximum = points.max(axis=0)
            cv2.rectangle(frame, tuple(minimum), tuple(maximum), color, 1)
        cv2.putText(frame, "Last hand check: " + decision.status, (12, frame.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
