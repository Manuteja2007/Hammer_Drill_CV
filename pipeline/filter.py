from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.signal import savgol_filter

@dataclass
class FilterResult:
    y_smooth: float
    velocity: float
    acceleration: float
    direction: str

class SavgolFilter:
    """Manages the Savitzky-Golay filtering window for a single coordinate (e.g. Y).
    Pixel values are converted to metres internally."""
    
    def __init__(self, buffer_size: int, poly_order: int, pixels_per_meter: float):
        self.buffer_size = buffer_size
        self.poly_order = poly_order
        self.pixels_per_meter = pixels_per_meter
        self.y_buffer = deque(maxlen=self.buffer_size)
        self.prev_y_smooth = None

    def reset(self):
        """Reset the filter state."""
        self.y_buffer.clear()
        self.prev_y_smooth = None

    def process(self, y_pixel: int, fps: float) -> Optional[FilterResult]:
        """Add a pixel measurement and return filtered metrics in metric units."""
        self.y_buffer.append(y_pixel)
        
        n = len(self.y_buffer)
        if n < 3:
            return None

        dt = 1.0 / fps
        win = n if n % 2 == 1 else n - 1
        win = max(win, 3)
        poly = min(self.poly_order, win - 1)

        try:
            # Convert pixel coords → metres before filtering
            y = np.array(self.y_buffer, dtype=np.float64) / self.pixels_per_meter
            
            y_s = float(savgol_filter(y, win, poly)[-1])
            v_y = float(savgol_filter(y, win, poly, deriv=1, delta=dt)[-1])
            a_y = float(savgol_filter(y, win, poly, deriv=2, delta=dt)[-1])
            
            # Direction inference (threshold in metres)
            direction = "→ Stable"
            if self.prev_y_smooth is not None:
                diff = y_s - self.prev_y_smooth
                if diff > 0.0002:
                    direction = "↓ Down"
                elif diff < -0.0002:
                    direction = "↑ Up"
            
            self.prev_y_smooth = y_s
            
            return FilterResult(
                y_smooth=y_s,
                velocity=v_y,
                acceleration=a_y,
                direction=direction
            )
            
        except Exception:
            return None
