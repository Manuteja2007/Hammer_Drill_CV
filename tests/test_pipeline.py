import unittest
from unittest.mock import patch

import numpy as np

from pipeline.detector import Detection
from pipeline.filter import FilterResult, SavgolFilter
from pipeline.hit_detector import HitDetector
from pipeline.hand_contact import HandContactGuard
from pipeline.processor import FrameProcessor
from state import AnalyticsState


class SequenceDetector:
    def __init__(self, points):
        self.points = iter(points)

    def detect(self, frame):
        point = next(self.points)
        return [] if point is None else [Detection((0, 0, 10, 10), point, 0.9, "golf_ball")]

    def get_target(self, detections):
        return detections[0] if detections else None


class HitDetectionTests(unittest.TestCase):
    def hit(self, detector, timestamp):
        detector.process(0.1, 0.1, 30, timestamp)
        return detector.process(0.1, -0.1, 30, timestamp + 0.03, 2.0)

    def test_configured_gates_reject_small_jitter_and_short_spikes(self):
        from pipeline.processor import make_hit_detector
        from unittest.mock import Mock
        validator = Mock(side_effect=AssertionError("Jitter must not trigger hand inference"))
        detector = make_hit_detector()
        for t, y, velocity in [(0, .1, .1), (.05, .101, .1), (.1, .1005, -.1)]:
            self.assertIsNone(detector.process(y, velocity, 30, t, candidate_validator=validator))
        for t, y, velocity in [(1, .1, .1), (1.01, .12, .1), (1.02, .115, -.1)]:
            self.assertIsNone(detector.process(y, velocity, 120, t, candidate_validator=validator))
        validator.assert_not_called()
        self.assertEqual(detector.get_total_hits(), 0)

    def test_configured_gates_accept_sustained_arc_and_ignore_rising_jump(self):
        from pipeline.processor import make_hit_detector
        detector = make_hit_detector()
        for t, y, velocity in [(0, .1, .1), (.05, .11, .1), (.1, .105, -.1)]:
            event = detector.process(y, velocity, 30, t)
        self.assertEqual(event.type, "HIT")
        # A tiny fall followed by a large tracking jump upward is not a fall excursion.
        for t, y, velocity in [(1, .1, .1), (1.05, .101, .1), (1.1, .05, -.5)]:
            event = detector.process(y, velocity, 30, t)
        self.assertIsNone(event)
        self.assertEqual(detector.get_total_hits(), 1)

    def test_positive_velocity_is_not_a_hit(self):
        detector = HitDetector()
        detector.process(0.1, 0.1, 30, 0)
        self.assertIsNone(detector.process(0.1, 0.005, 30, 0.03))
        self.assertEqual(detector.process(0.1, -0.1, 30, 0.06).type, "HIT")

    def test_velocity_dead_band_does_not_rearm_noise(self):
        detector = HitDetector()
        for index, value in enumerate([0.005, -0.005, 0.002, -0.02]):
            detector.process(0.1, value, 30, index / 30)
        self.assertEqual(detector.get_total_hits(), 0)

    def test_fall_and_acceleration_gates(self):
        detector = HitDetector(min_impact_acceleration=1.0)
        detector.process(0.1, 0.1, 30, 0, 0.1)
        self.assertIsNone(detector.process(0.1, -0.1, 30, 0.03, 0.1))
        self.assertEqual(self.hit(detector, 1).type, "HIT")

    def test_cooldown_uses_elapsed_time(self):
        detector = HitDetector(cooldown_frames=6)
        self.hit(detector, 0)
        self.assertIsNone(self.hit(detector, 0.05))
        self.assertEqual(self.hit(detector, 0.3).type, "HIT")

    def test_loss_cannot_create_a_reversal(self):
        detector = HitDetector()
        detector.process(0.1, 0.1, 30, 0)
        detector.process_lost_frame(30, 0.1)
        self.assertIsNone(detector.process(0.1, -0.1, 30, 0.2))

    def test_loss_timeout_uses_timestamp_not_lost_frame_count(self):
        detector = HitDetector()
        self.hit(detector, 0)
        result = detector.process_lost_frame(60, 4)
        self.assertEqual(result.reason, "ball_lost")
        self.assertIsNone(detector.process_lost_frame(60, 8))

    def test_timeout_precedes_late_new_hit(self):
        detector = HitDetector()
        self.hit(detector, 0)
        self.hit(detector, 4)
        self.assertEqual([event.type for event in detector.events], ["HIT", "MISS", "HIT"])
        self.assertEqual(detector.get_current_streak(), 1)

    def test_best_streak_includes_active_run(self):
        detector = HitDetector(cooldown_frames=0)
        self.hit(detector, 0)
        detector.process_lost_frame(30, 4)
        self.hit(detector, 5)
        self.hit(detector, 6)
        self.assertEqual(detector.get_summary()["best_streak"], 2)

    def test_finalize_is_idempotent_and_not_a_miss(self):
        detector = HitDetector()
        self.hit(detector, 0)
        detector.finalize_session(1)
        detector.finalize_session(1)
        self.assertEqual(detector.get_streak_history(), [1])
        self.assertEqual(detector.get_total_misses(), 0)
        self.assertEqual(detector.get_current_streak(), 0)
        self.assertEqual(detector.get_timeline_text(), "1 hit")


class MotionTests(unittest.TestCase):
    def test_irregular_samples_preserve_physical_derivatives(self):
        smoother = SavgolFilter(7, 2, 1000)
        for timestamp in [0.0, 0.03, 0.08, 0.2, 0.4, 0.45, 0.5]:
            y = 2 + 0.3 * timestamp + 0.5 * timestamp ** 2
            result = smoother.process(y * 1000, 30, timestamp)
        self.assertAlmostEqual(result.y_smooth, 2.275)
        self.assertAlmostEqual(result.velocity, 0.8)
        self.assertAlmostEqual(result.acceleration, 1.0)

    def test_nonmonotonic_timestamps_rebuffer(self):
        smoother = SavgolFilter(7, 2, 1000)
        for timestamp in [1, 2, 3]:
            smoother.process(100, 30, timestamp)
        self.assertIsNone(smoother.process(100, 30, 1))

    def test_averages_include_samples_beyond_old_cap(self):
        state = AnalyticsState()
        for _ in range(5000):
            state.record_motion(FilterResult(1, 1, 2, "Down"))
        for _ in range(5000):
            state.record_motion(FilterResult(1, 3, 4, "Down"))
        self.assertEqual(state.summary()["avg_velocity"], 2)
        self.assertEqual(state.summary()["avg_accel"], 3)

    def test_reset_clears_counts_metrics_and_timeline(self):
        state = AnalyticsState()
        state.stats.update(hit_count=7, timeline_text="old", time=12)
        state.record_motion(FilterResult(1, 1, 2, "Down"))
        state.reset(duration=20, fps=30, position=5)
        self.assertEqual(state.stats["hit_count"], 0)
        self.assertEqual(state.stats["timeline_text"], "No events yet")
        self.assertEqual(state.stats["time"], 5)
        self.assertEqual(state.summary()["avg_velocity"], 0)

    def test_shared_processor_supplies_acceleration_gate(self):
        detector = SequenceDetector([(1, 1)] * 4)
        processor = FrameProcessor(detector, hit_detector=HitDetector(min_impact_acceleration=1),
                                   hand_guard=HandContactGuard(enabled=False))
        frame = np.zeros((120, 240, 3), dtype=np.uint8)
        with patch.object(processor.filter, "process", side_effect=[
            FilterResult(1, 0.1, 0, "Down"), FilterResult(1, -0.1, 0, "Up"),
            FilterResult(1, 0.1, 2, "Down"), FilterResult(1, -0.1, 2, "Up"),
        ]):
            for timestamp in [0, 0.1, 1, 1.1]:
                processor.process(frame, 30, timestamp, annotate=False)
        self.assertEqual(processor.analytics.stats["hit_count"], 1)

    def test_lost_values_clear_and_long_gap_rebuffers(self):
        processor = FrameProcessor(SequenceDetector([(10, 10), (10, 20), (10, 30), None, (10, 40)]))
        frame = np.zeros((120, 240, 3), dtype=np.uint8)
        for timestamp in [0, 0.03, 0.06, 0.09]:
            processor.process(frame, 30, timestamp, annotate=False)
        self.assertEqual(processor.analytics.stats["v_y"], "--")
        result = processor.process(frame, 30, 2, annotate=False)
        self.assertIsNone(result.filtered)
        self.assertEqual(processor.analytics.stats["state"], "Buffering")


if __name__ == "__main__":
    unittest.main()
