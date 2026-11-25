"""Timestamp-aware local polynomial smoothing and derivatives."""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class FilterResult:
    y_smooth: float
    velocity: float
    acceleration: float
    direction: str


class SavgolFilter:
    """Savitzky-Golay least-squares fit, also valid for irregular samples.

    Actual timestamps prevent missed detections from becoming adjacent frames.
    Only measured positions contribute to motion analytics.
    """

    def __init__(self, buffer_size, poly_order, pixels_per_meter):
        if buffer_size < 3 or not 2 <= poly_order < buffer_size:
            raise ValueError("Use a window >= 3 and a polynomial degree >= 2 below it")
        if pixels_per_meter <= 0:
            raise ValueError("pixels_per_meter must be positive")
        self.poly_order = poly_order
        self.pixels_per_meter = pixels_per_meter
        self.samples = deque(maxlen=buffer_size)

    def reset(self):
        self.samples.clear()

    def process(self, y_pixel, fps, timestamp=None):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        if timestamp is None:
            timestamp = self.samples[-1][0] + 1 / fps if self.samples else 0.0
        if not math.isfinite(timestamp) or not math.isfinite(y_pixel):
            raise ValueError("Measurements must be finite")
        if self.samples and timestamp <= self.samples[-1][0]:
            self.reset()
        self.samples.append((timestamp, y_pixel / self.pixels_per_meter))
        if len(self.samples) < 3:
            return None
        samples = np.asarray(self.samples, dtype=np.float64)
        coefficients = np.polynomial.polynomial.polyfit(
            samples[:, 0] - timestamp, samples[:, 1], min(self.poly_order, len(samples) - 1)
        )
        velocity = float(coefficients[1])
        direction = "Down" if velocity > 0.001 else "Up" if velocity < -0.001 else "Stable"
        return FilterResult(float(coefficients[0]), velocity, float(2 * coefficients[2]), direction)
