"""Video metadata and deterministic, resource-safe frame iteration."""
import math
from pathlib import Path
import cv2


class VideoReader:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.cap = None

    def __enter__(self):
        if not self.path.is_file():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            self.cap.release()
            raise ValueError(f"Cannot open video: {self.path}")
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self

    def __iter__(self):
        index = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            index += 1
            yield index, index / self.fps, frame

    def __exit__(self, *exc):
        self.cap.release()
