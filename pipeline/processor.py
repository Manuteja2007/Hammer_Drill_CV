"""The shared per-frame pipeline used by HTTP streams and offline tools."""
from dataclasses import dataclass

import config
from state import AnalyticsState
from .annotator import Annotator
from .filter import SavgolFilter
from .hit_detector import HitDetector
from .hand_contact import HandContactGuard
from .tracker import KalmanTracker


def make_hit_detector(**overrides):
    parameters = {
        "cooldown_frames": config.HIT_COOLDOWN_FRAMES,
        "miss_timeout_seconds": config.MISS_TIMEOUT_SECONDS,
        "velocity_threshold": config.HIT_VELOCITY_THRESHOLD,
        "min_fall_velocity": config.HIT_MIN_FALL_VELOCITY,
        "min_fall_distance": config.HIT_MIN_FALL_DISTANCE,
        "min_fall_duration": config.HIT_MIN_FALL_DURATION,
        "min_impact_acceleration": config.HIT_MIN_IMPACT_ACCELERATION,
    }
    parameters.update(overrides)
    return HitDetector(**parameters)


@dataclass
class FrameResult:
    target: object
    filtered: object
    event: object
    hand_check: object = None


class FrameProcessor:
    def __init__(self, detector, analytics=None, hit_detector=None, hand_guard=None):
        self.detector = detector
        self.analytics = analytics if analytics is not None else AnalyticsState()
        self.tracker = KalmanTracker()
        self.filter = SavgolFilter(config.SG_BUFFER_SIZE, config.SG_POLY_ORDER, config.PIXELS_PER_METER)
        self.hit_detector = hit_detector if hit_detector is not None else make_hit_detector()
        self.hand_guard = hand_guard if hand_guard is not None else HandContactGuard()
        self.last_measurement_time = None
        self.flash_type = None
        self.flash_remaining = 0
        self.last_hand_overlay = None

    def reset(self, duration=0.0, fps=0.0, position=0.0):
        self.analytics.reset(duration, fps, position)
        self.tracker.reset()
        self.filter.reset()
        self.hit_detector.reset()
        self.hand_guard.reset()
        self.last_measurement_time = None
        self.flash_type = None
        self.flash_remaining = 0
        self.last_hand_overlay = None

    def process(self, frame, fps, timestamp, annotate=True):
        state = self.analytics
        state.stats["time"] = timestamp
        state.total_frames += 1
        detections = self.detector.detect(frame)
        target = self.detector.get_target(detections)
        prediction = self.tracker.predict()
        filtered = event = hand_check = None
        event_count = (len(self.hit_detector.events), len(self.hit_detector.rejections))
        # Cache unannotated local ball images. Model inference is deferred until a candidate tap.
        self.hand_guard.observe(frame, target, timestamp)

        def validate_candidate():
            nonlocal hand_check
            hand_check = self.hand_guard.validate()
            state.record_hand_check(hand_check)
            return hand_check
        if target is not None:
            state.detected_frames += 1
            if (self.last_measurement_time is not None
                    and timestamp - self.last_measurement_time > config.MAX_MEASUREMENT_GAP_SECONDS):
                self.filter.reset()
                self.hit_detector.reset_motion()
                self.tracker.reset()
                prediction = None
            self.last_measurement_time = timestamp
            self.tracker.update(target.center)
            state.stats.update(ball_x=round(target.center[0] / config.PIXELS_PER_METER, 6),
                               ball_y=round(target.center[1] / config.PIXELS_PER_METER, 6),
                               confidence=round(target.confidence, 4))
            filtered = self.filter.process(target.center[1], fps, timestamp)
            if filtered is not None:
                state.record_motion(filtered)
                state.stats["state"] = "Tracking"
                event = self.hit_detector.process(filtered.y_smooth, filtered.velocity, fps,
                                                  timestamp, filtered.acceleration, validate_candidate)
                if event is not None and event.type == "REJECTED":
                    state.stats["state"] = hand_check.status
            else:
                state.stats["state"] = "Buffering"
                event = self.hit_detector.process_lost_frame(fps, timestamp)
        else:
            state.lost_frames += 1
            self.tracker.mark_lost(prediction)
            state.stats["state"] = "Lost/Predicting"
            for key in ("ball_x", "ball_y", "confidence", "direction", "v_y", "accel_y", "y_smooth"):
                state.stats[key] = "--"
            event = self.hit_detector.process_lost_frame(fps, timestamp)
        if (len(self.hit_detector.events), len(self.hit_detector.rejections)) != event_count:
            state.sync_hits(self.hit_detector)
        if event is not None:
            self.flash_type, self.flash_remaining = event.type, 8
        if hand_check is not None:
            self.last_hand_overlay = hand_check
        if target is None:
            self.last_hand_overlay = None
        if annotate:
            Annotator.draw_detections(frame, detections)
            Annotator.draw_tracker(frame, self.tracker, prediction)
            # Show recent evidence briefly, labelled as a past check. It never
            # replaces fresh validation at a later candidate or triggers inference.
            if self.last_hand_overlay is not None and 0 <= timestamp - self.last_hand_overlay.timestamp <= 0.20:
                Annotator.draw_hand_check(frame, self.last_hand_overlay)
            Annotator.draw_hit_miss_overlay(
                frame, self.flash_type, self.flash_remaining,
                self.hit_detector.get_current_streak(), state.stats["hit_count"], state.stats["miss_count"]
            )
        self.flash_remaining = max(0, self.flash_remaining - 1)
        return FrameResult(target, filtered, event, hand_check)

    def finalize(self):
        self.hit_detector.finalize_session(self.analytics.stats["time"])
        self.analytics.sync_hits(self.hit_detector)
        self.hand_guard.reset()
        self.last_hand_overlay = None
