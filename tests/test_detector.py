from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from pipeline.detector import YOLODetector


class DetectorTests(unittest.TestCase):
    def test_only_ball_class_is_requested_and_nonball_outputs_are_discarded(self):
        detector = object.__new__(YOLODetector)
        detector.names = {0: "golf_ball", 1: "hammer"}
        detector.target_class = "golf_ball"
        detector.target_ids = [0]
        detector.conf_threshold = .2
        detector._lock = Lock()
        def box(cls, confidence):
            return SimpleNamespace(cls=[cls], conf=[confidence], xyxy=[[10, 20, 30, 40]])
        detector.model = Mock(return_value=[SimpleNamespace(boxes=[
            box(1, .99), box(0, .3), box(0, .9), box(0, .1)])])
        detections = detector.detect(np.zeros((60, 60, 3), dtype=np.uint8))
        self.assertEqual(detector.model.call_args.kwargs["classes"], [0])
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "golf_ball")
        self.assertEqual(detections[0].confidence, .9)
        self.assertEqual(detections[0].center, (20, 30))
