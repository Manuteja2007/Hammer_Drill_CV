"""Check likely hand-ball contact only when the motion detector proposes a hit.

MediaPipe's palm/landmark models run through OpenCV DNN. No runtime downloads or
package installation. Geometry is image-space evidence, not a 3D touch sensor.
"""

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter

import cv2
import numpy as np

import config
from .detector import Detection

FINGER_CHAINS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
                 (13, 14, 15, 16), (17, 18, 19, 20))
PALM_INDICES = (0, 1, 5, 9, 13, 17)


@dataclass
class Hand:
    landmarks: np.ndarray
    confidence: float


@dataclass
class HandResult:
    hands: list = field(default_factory=list)
    uncertain: bool = False


@dataclass
class ContactDecision:
    allowed: bool
    contact: bool
    reason: str
    status: str
    timestamp: float
    latency_ms: float = 0.0
    sample_timestamp: float = 0.0
    hands: list = field(default_factory=list)


def ball_touches_hand(ball, hand):
    """Ball circle vs palm polygon and finger capsules, not the whole hand box."""
    points = np.asarray(hand.landmarks, dtype=np.float32)
    if points.shape != (21, 2) or not np.isfinite(points).all():
        raise ValueError("Expected 21 finite hand landmarks")
    center = np.asarray(ball.center, dtype=np.float32)
    x1, y1, x2, y2 = ball.box
    ball_radius = max(1.0, (max(0, x2 - x1) + max(0, y2 - y1)) / 4)
    margin = config.HAND_CONTACT_MARGIN_PIXELS
    palm = cv2.convexHull(points[list(PALM_INDICES)])
    if cv2.pointPolygonTest(palm, (float(center[0]), float(center[1])), True) >= -(ball_radius + margin):
        return True
    palm_width = float(np.linalg.norm(points[5] - points[17]))
    distance_limit = ball_radius + margin + palm_width * config.HAND_FINGER_RADIUS_PALM_RATIO
    for chain in FINGER_CHAINS:
        for start, end in zip(chain, chain[1:]):
            a, b = points[start], points[end]
            vector = b - a
            squared_length = float(vector @ vector)
            fraction = np.clip(float((center - a) @ vector) / squared_length, 0, 1) if squared_length else 0
            closest = a + fraction * vector
            if float(np.linalg.norm(center - closest)) <= distance_limit:
                return True
    return False


class ONNXNet:
    """Minimal inference interface for the vendor transforms, single-thread CPU."""

    def __init__(self, path, runtime):
        options = runtime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = runtime.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.session = runtime.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise RuntimeError(f"Could not load hand model {path.name}") from exc
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input = None

    def setInput(self, blob):
        self.input = np.ascontiguousarray(blob, dtype=np.float32)

    def getUnconnectedOutLayersNames(self):
        return self.output_names

    def forward(self, names):
        try:
            return self.session.run(list(names), {self.input_name: self.input})
        except Exception as exc:
            raise RuntimeError("Hand inference failed") from exc


class MediaPipeHandDetector:
    """Shared model weights with serialized predictor access between sessions."""

    def __init__(self):
        for path in (config.HAND_PALM_MODEL_PATH, config.HAND_LANDMARK_MODEL_PATH):
            if not path.is_file():
                raise FileNotFoundError(f"Missing hand model: {path.name}")
        from third_party.opencv_zoo.mp_palmdet import MPPalmDet
        from third_party.opencv_zoo.mp_handpose import MPHandPose

        try:
            import onnxruntime
        except ImportError:
            onnxruntime = None
        palm_net = ONNXNet(config.HAND_PALM_MODEL_PATH, onnxruntime) if onnxruntime else None
        pose_net = ONNXNet(config.HAND_LANDMARK_MODEL_PATH, onnxruntime) if onnxruntime else None
        self.backend = "onnxruntime_cpu_1thread" if onnxruntime else "opencv_dnn_cpu"
        self.palm = MPPalmDet(str(config.HAND_PALM_MODEL_PATH), scoreThreshold=config.HAND_PALM_CONFIDENCE, model=palm_net)
        self.pose = MPHandPose(str(config.HAND_LANDMARK_MODEL_PATH), confThreshold=config.HAND_LANDMARK_CONFIDENCE, model=pose_net)
        self.lock = Lock()

    def detect(self, image, ball):
        with self.lock:
            palms = self.palm.infer(image)
            nearby = []
            center = np.asarray(ball.center)
            for palm in palms:
                low, high = palm[:2], palm[2:4]
                distance = float(np.linalg.norm(center - np.clip(center, low, high)))
                if distance <= max(high - low) * config.HAND_PALM_REACH_FACTOR:
                    nearby.append((distance, palm))
            nearby.sort(key=lambda item: item[0])
            result = HandResult(uncertain=len(nearby) > config.HAND_MAX_NEARBY_HANDS)
            for _, palm in nearby[:config.HAND_MAX_NEARBY_HANDS]:
                try:
                    pose = self.pose.infer(image, palm)
                except cv2.error:
                    result.uncertain = True
                    continue
                if pose is None:
                    result.uncertain = True
                else:
                    points = pose[4:67].reshape(21, 3)[:, :2]
                    if not np.isfinite(points).all() or not np.isfinite(pose[-1]):
                        result.uncertain = True
                    else:
                        result.hands.append(Hand(points.copy(), float(pose[-1])))
            return result


_detector = None
_model_lock = Lock()


def get_hand_detector():
    global _detector
    with _model_lock:
        if _detector is None:
            _detector = MediaPipeHandDetector()
        return _detector


@dataclass
class ContactFrame:
    image: np.ndarray
    ball: Detection
    timestamp: float
    offset: tuple


class HandContactGuard:
    def __init__(self, detector_factory=get_hand_detector, enabled=None):
        self.detector_factory = detector_factory
        self.enabled = config.HAND_CHECK_ENABLED if enabled is None else enabled
        self.history = deque(maxlen=config.HAND_HISTORY_FRAMES)
        self.current_decision = None

    def reset(self):
        self.history.clear()
        self.current_decision = None

    def observe(self, frame, ball, timestamp):
        self.current_decision = None
        if not self.enabled:
            return
        if ball is None or (self.history and timestamp <= self.history[-1].timestamp):
            self.history.clear()
        if ball is None:
            return
        while self.history and timestamp - self.history[0].timestamp > config.HAND_CONTACT_LOOKBACK_SECONDS:
            self.history.popleft()
        x1, y1, x2, y2 = ball.box
        side = max(config.HAND_ROI_MIN_SIDE, max(x2 - x1, y2 - y1) * config.HAND_ROI_BALL_DIAMETERS)
        height, width = frame.shape[:2]
        left = max(0, int(ball.center[0] - side / 2))
        top = max(0, int(ball.center[1] - side / 2))
        right = min(width, int(ball.center[0] + side / 2))
        bottom = min(height, int(ball.center[1] + side / 2))
        if right <= left or bottom <= top:
            self.history.clear()
            return
        local_ball = Detection((x1 - left, y1 - top, x2 - left, y2 - top),
                               (ball.center[0] - left, ball.center[1] - top), ball.confidence, ball.label)
        self.history.append(ContactFrame(frame[top:bottom, left:right].copy(), local_ball, timestamp, (left, top)))

    def validate(self):
        if self.current_decision is not None:
            return self.current_decision
        timestamp = self.history[-1].timestamp if self.history else 0.0
        if not self.enabled:
            return ContactDecision(True, False, "disabled", "Disabled", timestamp)
        started = perf_counter()
        try:
            if not self.history:
                raise ValueError("No current ball image to verify")
            detector = self.detector_factory()
            latest = self.history[-1]
            # Current frame plus one lookback frame: bounded latency and no future-frame delay.
            samples = [latest]
            earlier = self.history[0]
            if latest.timestamp - earlier.timestamp >= 0.025:
                samples.append(earlier)
            uncertain, display_hands, contact_time = False, [], timestamp
            contact = False
            for sample in samples:
                result = detector.detect(sample.image, sample.ball)
                uncertain |= result.uncertain
                if sample is latest:
                    display_hands = [Hand(hand.landmarks + np.asarray(sample.offset), hand.confidence)
                                     for hand in result.hands]
                if any(ball_touches_hand(sample.ball, hand) for hand in result.hands):
                    contact, contact_time = True, sample.timestamp
                    break
            allowed = not contact and not (uncertain and config.HAND_REJECT_UNCERTAIN)
            reason = "hand_contact" if contact else "hand_uncertain" if uncertain else "clear"
            status = ("Hand contact - tap ignored" if contact else "Hand unclear - tap unverified" if not allowed
                      else "Hand unclear - no contact confirmed" if uncertain
                      else "Clear at last tap check")
            decision = ContactDecision(allowed, contact, reason, status, timestamp,
                                       sample_timestamp=contact_time, hands=display_hands)
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            decision = ContactDecision(False, False, "hand_check_unavailable",
                                       f"Hand check unavailable: {exc}", timestamp)
        decision.latency_ms = (perf_counter() - started) * 1000
        self.current_decision = decision
        return decision
