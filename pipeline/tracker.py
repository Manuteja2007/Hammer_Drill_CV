from collections import deque
from typing import Tuple, Optional
import numpy as np
import cv2

class KalmanTracker:
    """Encapsulates Kalman Filter logic for 2D coordinate tracking."""
    
    def __init__(self, max_trail: int = 30):
        self.max_trail = max_trail
        self.trail = deque(maxlen=self.max_trail)
        self.initialised = False
        self.kf = self._init_kalman()

    def _init_kalman(self):
        """Create and configure a 4-state / 2-measurement Kalman filter."""
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],
                                           [0,0,1,0],[0,0,0,1]], np.float32)
        kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        return kf

    def reset(self):
        """Reset the tracker state."""
        self.trail.clear()
        self.initialised = False
        self.kf = self._init_kalman()

    def predict(self) -> Optional[Tuple[int, int]]:
        """Predict the next state. Returns predicted (x, y) if initialised."""
        if not self.initialised:
            return None
            
        pred = self.kf.predict()
        px, py = int(pred[0].item()), int(pred[1].item())
        return (px, py)

    def update(self, center: Tuple[int, int]):
        """Update the Kalman filter with a new measurement and append to trail."""
        if not self.initialised:
            m = np.array([[np.float32(center[0])],
                          [np.float32(center[1])]])
            self.kf.statePre = np.array(
                [[m[0][0]], [m[1][0]], [0], [0]], np.float32
            )
            self.initialised = True
            
        self.kf.correct(np.array(
            [[np.float32(center[0])],
             [np.float32(center[1])]]
        ))
        self.trail.append(center)

    def mark_lost(self, predicted_pos: Optional[Tuple[int, int]]):
        """Handle a frame where detection was lost. Append prediction to trail."""
        if self.initialised and predicted_pos is not None:
            self.trail.append(predicted_pos)
