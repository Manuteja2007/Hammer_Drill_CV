import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pipeline.detector import Detection
from pipeline.filter import FilterResult
from pipeline.hand_contact import ContactDecision, Hand, HandContactGuard, HandResult, ONNXNet, ball_touches_hand
from pipeline.hit_detector import HitDetector
from pipeline.processor import FrameProcessor
from scripts.visualise_hit_sweep import run_detector_on_data


def hand_points():
    return np.array([
        [50, 120], [30, 105], [20, 90], [10, 80], [0, 70],
        [35, 90], [30, 65], [27, 40], [25, 15],
        [50, 85], [50, 55], [50, 30], [50, 5],
        [65, 90], [70, 65], [73, 40], [75, 15],
        [78, 100], [90, 80], [98, 65], [105, 50],
    ], dtype=np.float32)


def ball_at(x, y, radius=3):
    return Detection((x - radius, y - radius, x + radius, y + radius), (x, y), 0.95, "golf_ball")


class GeometryTests(unittest.TestCase):
    def test_fingertip_and_middle_of_finger_contact(self):
        hand = Hand(hand_points(), 0.95)
        self.assertTrue(ball_touches_hand(ball_at(25, 15), hand))
        self.assertTrue(ball_touches_hand(ball_at(28, 50), hand))

    def test_palm_contact(self):
        self.assertTrue(ball_touches_hand(ball_at(50, 105), Hand(hand_points(), 0.95)))

    def test_empty_space_inside_hand_bbox_is_not_contact(self):
        hand = Hand(hand_points(), 0.95)
        self.assertFalse(ball_touches_hand(ball_at(10, 20), hand))
        self.assertFalse(ball_touches_hand(ball_at(38, 35), hand))

    def test_distant_hammer_gripping_hand_does_not_block_ball(self):
        self.assertFalse(ball_touches_hand(ball_at(50, 105), Hand(hand_points() + 200, 0.95)))

    def test_invalid_landmarks_are_not_silently_clear(self):
        with self.assertRaises(ValueError):
            ball_touches_hand(ball_at(10, 10), Hand(np.full((21, 2), np.nan), 0.9))


class BackendFailureTests(unittest.TestCase):
    def test_runtime_specific_model_error_becomes_a_hand_check_error(self):
        runtime = Mock()
        runtime.InferenceSession.side_effect = Exception("invalid model protobuf")
        with self.assertRaisesRegex(RuntimeError, "Could not load hand model broken.onnx"):
            ONNXNet(Path("broken.onnx"), runtime)

    def test_runtime_specific_inference_error_becomes_a_hand_check_error(self):
        runtime = Mock()
        session = runtime.InferenceSession.return_value
        session.get_inputs.return_value = [SimpleNamespace(name="input_1")]
        session.get_outputs.return_value = [SimpleNamespace(name="Identity")]
        session.run.side_effect = Exception("backend inference failed")
        network = ONNXNet(Path("model.onnx"), runtime)
        network.setInput(np.zeros((1, 192, 192, 3), dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "Hand inference failed"):
            network.forward(["Identity"])


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((400, 400, 3), dtype=np.uint8)
        self.ball = ball_at(200, 200)

    def detector(self, touching=False, uncertain=False):
        detector = Mock()
        def detect(image, ball):
            points = hand_points() + np.asarray(ball.center) - [50, 105]
            if not touching:
                points += 180
            return HandResult([Hand(points, 0.95)], uncertain)
        detector.detect.side_effect = detect
        return detector

    def test_observing_frames_never_loads_or_runs_the_hand_models(self):
        factory = Mock(side_effect=AssertionError("Unexpected model work"))
        guard = HandContactGuard(factory)
        for timestamp in [0, 0.03, 0.06]:
            guard.observe(self.frame, self.ball, timestamp)
        factory.assert_not_called()

    def test_contact_rejected_and_decision_cached_per_frame(self):
        detector = self.detector(touching=True)
        guard = HandContactGuard(lambda: detector)
        guard.observe(self.frame, self.ball, 0)
        decision = guard.validate()
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.contact)
        self.assertEqual(decision.reason, "hand_contact")
        self.assertIs(guard.validate(), decision)
        self.assertEqual(detector.detect.call_count, 1)
        # Cropped coordinates are mapped back for the overlay.
        np.testing.assert_allclose(decision.hands[0].landmarks[9], [200, 180])

    def test_recent_contact_is_rejected_even_after_hand_releases_ball(self):
        detector = self.detector(touching=False)
        original = detector.detect.side_effect
        def detect(image, ball):
            if image[0, 0, 0] == 1:
                return HandResult([Hand(hand_points() + np.asarray(ball.center) - [50, 105], 0.95)])
            return original(image, ball)
        detector.detect.side_effect = detect
        guard = HandContactGuard(lambda: detector)
        guard.observe(np.ones_like(self.frame), self.ball, 0)
        guard.observe(self.frame, self.ball, 0.08)
        decision = guard.validate()
        self.assertTrue(decision.contact)
        self.assertEqual(decision.sample_timestamp, 0)
        self.assertEqual(detector.detect.call_count, 2)

    def test_old_contact_does_not_block_later_genuine_tap(self):
        detector = self.detector(touching=False)
        guard = HandContactGuard(lambda: detector)
        guard.observe(self.frame, self.ball, 0)
        guard.observe(self.frame, self.ball, 1)
        self.assertTrue(guard.validate().allowed)
        self.assertEqual(detector.detect.call_count, 1)

    def test_uncertain_nearby_hand_is_unverified_not_proven_contact(self):
        detector = self.detector(touching=False, uncertain=True)
        guard = HandContactGuard(lambda: detector)
        guard.observe(self.frame, self.ball, 0)
        with patch("config.HAND_REJECT_UNCERTAIN", True):
            decision = guard.validate()
        self.assertFalse(decision.allowed)
        self.assertFalse(decision.contact)
        self.assertEqual(decision.reason, "hand_uncertain")

    def test_weak_hand_pose_alone_is_not_contact_in_default_mode(self):
        detector = self.detector(touching=False, uncertain=True)
        guard = HandContactGuard(lambda: detector)
        guard.observe(self.frame, self.ball, 0)
        decision = guard.validate()
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.contact)
        self.assertEqual(decision.reason, "hand_uncertain")

    def test_missing_model_cannot_silently_disable_rejection(self):
        guard = HandContactGuard(Mock(side_effect=FileNotFoundError("missing.onnx")))
        guard.observe(self.frame, self.ball, 0)
        decision = guard.validate()
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "hand_check_unavailable")

    def test_no_hand_detection_allows_motion_candidate(self):
        detector = Mock()
        detector.detect.return_value = HandResult()
        guard = HandContactGuard(lambda: detector)
        guard.observe(self.frame, self.ball, 0)
        self.assertTrue(guard.validate().allowed)

    def test_loss_reset_and_backward_seek_remove_history(self):
        guard = HandContactGuard()
        guard.observe(self.frame, self.ball, 1)
        guard.observe(self.frame, self.ball, 0)
        self.assertEqual(len(guard.history), 1)
        guard.observe(self.frame, None, 0.1)
        self.assertFalse(guard.history)
        guard.observe(self.frame, self.ball, 2)
        guard.reset()
        self.assertFalse(guard.history)
        self.assertIsNone(guard.current_decision)


class HitGateTests(unittest.TestCase):
    def decision(self, allowed, reason="hand_contact"):
        return ContactDecision(allowed, not allowed, reason, "test", 0.03, sample_timestamp=0.03)

    def test_rejected_candidate_never_enters_hits_streak_or_cooldown(self):
        detector = HitDetector()
        detector.process(1, 0.1, 30, 0)
        event = detector.process(1, -0.1, 30, 0.03, candidate_validator=lambda: self.decision(False))
        self.assertEqual(event.type, "REJECTED")
        self.assertEqual(detector.get_total_hits(), 0)
        self.assertEqual(detector.get_current_streak(), 0)
        self.assertEqual(detector.get_summary()["hand_rejected_count"], 1)
        self.assertEqual(detector.get_timeline(), [])
        detector.process(1, 0.1, 30, 0.05)
        genuine = detector.process(1, -0.1, 30, 0.08, candidate_validator=lambda: self.decision(True))
        self.assertEqual(genuine.type, "HIT")
        self.assertEqual(genuine.streak_position, 1)

    def test_existing_streak_is_not_incremented_or_retroactively_modified(self):
        detector = HitDetector(cooldown_frames=0)
        detector.process(1, 0.1, 30, 0)
        detector.process(1, -0.1, 30, 0.03)
        detector.process(1, 0.1, 30, 0.1)
        detector.process(1, -0.1, 30, 0.13, candidate_validator=lambda: self.decision(False))
        self.assertEqual(detector.get_total_hits(), 1)
        self.assertEqual(detector.get_current_streak(), 1)
        detector.finalize_session()
        self.assertEqual(detector.get_summary()["streak_history"], [1])

    def test_non_candidates_do_not_invoke_validator(self):
        validator = Mock(side_effect=AssertionError("Not a candidate"))
        detector = HitDetector()
        for timestamp, velocity in [(0, 0.1), (0.03, 0.005), (0.06, 0.001)]:
            detector.process(1, velocity, 30, timestamp, candidate_validator=validator)
        validator.assert_not_called()

    def test_unclear_accepted_taps_are_tagged_for_review(self):
        detector = HitDetector()
        detector.process(1, 0.1, 30, 0)
        decision = ContactDecision(True, False, "hand_uncertain", "Hand unclear", 0.03)
        event = detector.process(1, -0.1, 30, 0.03, candidate_validator=lambda: decision)
        self.assertEqual(event.type, "HIT")
        self.assertEqual(event.hand_check, "hand_uncertain")
        self.assertEqual(detector.get_summary()["uncertain_hit_count"], 1)

    def test_shared_processor_exposes_rejection_and_timing(self):
        detector = Mock()
        detector.detect.return_value = [ball_at(200, 200)]
        detector.get_target.side_effect = lambda detections: detections[0]
        guard = Mock()
        guard.validate.return_value = ContactDecision(False, True, "hand_contact", "Hand contact - tap ignored",
                                                      0.03, latency_ms=8.5, sample_timestamp=0.03)
        processor = FrameProcessor(detector, hand_guard=guard)
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        with patch.object(processor.filter, "process", side_effect=[
            FilterResult(1, 0.1, 2, "Down"), FilterResult(1.01, 0.1, 2, "Down"), FilterResult(1.005, -0.1, 2, "Up")
        ]):
            processor.process(frame, 30, 0, annotate=False)
            processor.process(frame, 30, 0.05, annotate=False)
            result = processor.process(frame, 30, 0.1, annotate=False)
        self.assertEqual(result.event.type, "REJECTED")
        stats = processor.analytics.stats
        self.assertEqual(stats["hit_count"], 0)
        self.assertEqual(stats["rejected_count"], 1)
        self.assertEqual(stats["last_event"], "REJECTED")
        self.assertEqual(stats["hand_check_ms"], 8.5)
        self.assertEqual(processor.analytics.summary()["avg_hand_check_ms"], 8.5)
        processor.reset()
        self.assertEqual(processor.analytics.stats["rejected_count"], 0)
        self.assertEqual(processor.analytics.hand_check_count, 0)
        guard.reset.assert_called_once()

    def test_sweep_applies_the_same_cached_contact_gate(self):
        rows = [dict(timestamp=t, y_smooth=y, velocity=v, acceleration=2,
                     contact_allowed=allowed, contact_reason="clear" if allowed else "hand_contact",
                     contact_sample_timestamp=t)
                for t, y, v, allowed in [(0, 1, 0.1, True), (0.05, 1.01, 0.1, True), (0.1, 1.005, -0.1, False), (0.3, 1, 0.1, True), (0.35, 1.01, 0.1, True), (0.4, 1.005, -0.1, True)]]
        summary = run_detector_on_data(rows, 30, {})
        self.assertEqual(summary["total_hits"], 1)
        self.assertEqual(summary["hand_rejected_count"], 1)

    def test_overlay_hold_does_not_run_extra_inference_or_survive_reset(self):
        detector = Mock()
        detector.detect.return_value = [ball_at(200, 200)]
        detector.get_target.side_effect = lambda detections: detections[0]
        guard = Mock()
        guard.validate.return_value = ContactDecision(True, False, "clear", "Clear", 0.1)
        processor = FrameProcessor(detector, hand_guard=guard)
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        with patch.object(processor.filter, "process", side_effect=[
            FilterResult(1, 0.1, 2, "Down"), FilterResult(1.01, 0.1, 2, "Down"), FilterResult(1.005, -0.1, 2, "Up"),
            FilterResult(1, -0.1, 2, "Up"), FilterResult(1, -0.1, 2, "Up")
        ]), patch("pipeline.processor.Annotator.draw_hand_check") as draw:
            for timestamp in [0, 0.05, 0.1, 0.15, 0.4]:
                processor.process(frame, 30, timestamp)
            self.assertEqual(guard.validate.call_count, 1)
            self.assertEqual(draw.call_count, 2)
        processor.reset()
        self.assertIsNone(processor.last_hand_overlay)


if __name__ == "__main__":
    unittest.main()
