from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional

from threading import Lock

@dataclass
class Detection:
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]         # (cx, cy)
    confidence: float
    label: str

class BaseDetector(ABC):
    """Abstract base class for object detectors. Makes it easy to plug in new models."""

    @abstractmethod
    def detect(self, frame) -> List[Detection]:
        """Process a frame and return a list of detections."""
        pass

class YOLODetector(BaseDetector):
    """YOLOv8 implementation of the BaseDetector."""

    def __init__(self, model_path: str, conf_threshold: float, target_class: str):
        self.conf_threshold = conf_threshold
        self.target_class = target_class
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.names = self.model.names
        self.target_ids = [key for key, value in self.names.items() if value == target_class]
        if not self.target_ids:
            raise ValueError(f"Model has no {target_class!r} class")
        self._lock = Lock()

    def detect(self, frame) -> List[Detection]:
        with self._lock:
            results = self.model(frame, conf=self.conf_threshold, classes=self.target_ids, verbose=False)[0]
        best_per_cls = self._best_detections(results)

        detections = []
        for cls_id, (box, conf) in best_per_cls.items():
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.names.get(cls_id, str(cls_id))
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            detections.append(Detection(
                box=(x1, y1, x2, y2),
                center=center,
                confidence=conf,
                label=label
            ))

        return detections

    def _best_detections(self, results) -> dict:
        """Keep only the highest-confidence box per class (above threshold)."""
        best = {}
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if self.names.get(cls) != self.target_class or conf < self.conf_threshold:
                continue
            if cls not in best or conf > best[cls][1]:
                best[cls] = (box, conf)
        return best

    def get_target(self, detections: List[Detection]) -> Optional[Detection]:
        """Extract the highest confidence detection matching the target class."""
        best_target = None
        for d in detections:
            if d.label == self.target_class:
                if best_target is None or d.confidence > best_target.confidence:
                    best_target = d
        return best_target
