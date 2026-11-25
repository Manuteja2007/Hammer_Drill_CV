import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app import create_app
from stream import StreamController, StreamError


class EmptyDetector:
    def detect(self, frame):
        return []

    def get_target(self, detections):
        return None


class FakeCapture:
    instances = []

    def __init__(self, source):
        self.source = source
        self.index = 0
        self.released = False
        self.instances.append(self)

    def isOpened(self):
        return not self.released

    def get(self, key):
        return {cv2.CAP_PROP_FPS: 29.97, cv2.CAP_PROP_FRAME_COUNT: 3}.get(key, 0)

    def set(self, key, value):
        self.index = int(value)
        return True

    def read(self):
        if self.index >= 3:
            return False, None
        self.index += 1
        return True, np.zeros((120, 240, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.stream = StreamController("video", lambda: EmptyDetector(), FakeCapture)
        self.token = self.stream.open("sample.mp4")
        self.stream.wake.wait = lambda timeout: False

    def tearDown(self):
        self.stream.stop()

    def test_eof_preserves_position_releases_capture_and_replays_cleanly(self):
        generator = self.stream.generate(self.token)
        cap = self.stream.cap
        for _ in range(4):
            next(generator)
        self.assertTrue(cap.released)
        self.assertEqual(self.stream.snapshot()["state"], "Video Ended")
        self.assertAlmostEqual(self.stream.snapshot()["time"], 3 / 29.97)
        self.assertEqual(self.stream.summary()["total_misses"], 0)
        self.stream.control("play")
        next(generator)
        self.assertEqual(self.stream.summary()["total_frames"], 1)
        generator.close()
        self.assertFalse(self.stream.connected)

    def test_disconnect_releases_capture(self):
        cap = self.stream.cap
        generator = self.stream.generate(self.token)
        next(generator)
        generator.close()
        self.assertTrue(cap.released)
        self.assertFalse(self.stream.connected)

    def test_stop_finalizes_before_summary(self):
        detector = self.stream.processor.hit_detector
        detector.process(1, 0.1, 30, 0)
        detector.process(1.01, 0.1, 30, 0.05)
        detector.process(1.005, -0.1, 30, 0.1)
        self.stream.control("stop")
        summary = self.stream.summary()
        self.assertEqual(summary["total_hits"], 1)
        self.assertEqual(summary["streaks"], [1])
        self.assertEqual(summary["total_misses"], 0)

    def test_seek_clears_stale_metrics(self):
        self.stream.analytics.stats.update(hit_count=8, v_y=3, timeline_text="stale")
        self.stream.control("seek", 0.05)
        self.assertEqual(self.stream.snapshot()["hit_count"], 0)
        self.assertEqual(self.stream.snapshot()["v_y"], "--")
        self.assertEqual(self.stream.snapshot()["timeline_text"], "No events yet")
        self.assertGreater(self.stream.snapshot()["duration"], 0)

    def test_duplicate_connection_rejected(self):
        with self.assertRaises(StreamError) as raised:
            self.stream.open("other.mp4")
        self.assertEqual(raised.exception.status, 409)

    def test_unattached_preflight_capture_expires(self):
        cap = self.stream.cap
        self.stream.prepared_at -= 31
        self.stream.expire_prepared()
        self.assertTrue(cap.released)
        self.assertFalse(self.stream.connected)

    def test_processing_failure_releases_capture(self):
        cap = self.stream.cap
        with patch.object(self.stream.processor, "process", side_effect=RuntimeError("inference failed")):
            with self.assertRaises(RuntimeError):
                next(self.stream.generate(self.token))
        self.assertTrue(cap.released)
        self.assertFalse(self.stream.connected)

    def test_capture_open_failure_releases_capture(self):
        stream = StreamController("video", lambda: EmptyDetector(), FakeCapture)
        with patch.object(FakeCapture, "isOpened", return_value=False):
            with self.assertRaises(StreamError):
                stream.open("broken.mp4")
        self.assertTrue(FakeCapture.instances[-1].released)

    def test_fractional_frame_rate_preserved_and_paced(self):
        self.assertEqual(self.stream.fps, 29.97)
        with patch.object(self.stream.wake, "wait") as wait:
            generator = self.stream.generate(self.token)
            next(generator)
            self.assertGreater(wait.call_args.args[0], 0)
            generator.close()

    def test_live_reconnect_does_not_seek_the_webcam(self):
        live = StreamController("live", lambda: EmptyDetector(), FakeCapture)
        token = live.open(0)
        live.wake.wait = lambda timeout: False
        generator = live.generate(token)
        try:
            for _ in range(4):
                next(generator)
            self.assertEqual(live.snapshot()["state"], "Camera Disconnected")
            with patch.object(FakeCapture, "set", side_effect=AssertionError("Webcam cannot seek")):
                live.control("play")
                next(generator)
            self.assertEqual(live.summary()["total_frames"], 1)
        finally:
            generator.close()
            live.stop()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.app = create_app(lambda: EmptyDetector(), FakeCapture,
                              test_config={"TESTING": True, "UPLOAD_FOLDER": self.directory.name})
        self.client = self.app.test_client()

    def tearDown(self):
        for browser in self.app.extensions["tracker_sessions"].sessions.values():
            browser.close()
        self.directory.cleanup()

    def upload(self, client, name="clip.mp4"):
        return client.post("/upload_video", data={"video": (io.BytesIO(b"video"), name)})

    def test_preflight_start_and_single_consumer_attachment(self):
        self.upload(self.client)
        started = self.client.post("/start_stream/video")
        self.assertEqual(started.status_code, 200)
        url = started.get_json()["feed_url"]
        self.assertIn("token=", url)
        first = self.client.get(url, buffered=False)
        self.assertEqual(first.status_code, 200)
        duplicate = self.client.get(url)
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(self.client.post("/start_stream/video").status_code, 409)
        self.assertEqual(self.client.post("/video_control", json={"mode":"video", "action":"stop"}).status_code, 200)
        self.assertEqual(self.client.get(url).status_code, 409)
        first.close()

    def test_preflight_errors_are_json_before_mjpeg_starts(self):
        self.assertEqual(self.client.post("/start_stream/video").status_code, 404)
        self.assertEqual(self.client.post("/start_stream/unknown").status_code, 400)
        self.assertEqual(self.client.get("/video_feed_live?token=expired").status_code, 409)

    def test_new_interface_names_and_no_store_for_session_data(self):
        page = self.client.get("/")
        self.assertIn(b"Set up a drill session", page.data)
        self.assertIn(b"Session notes", page.data)
        self.assertNotIn(b"fonts.googleapis.com", page.data)
        self.assertEqual(page.headers["Cache-Control"], "no-store")
        self.assertEqual(self.client.get("/session_summary/video").headers["Cache-Control"], "no-store")

    def test_index_and_assets(self):
        for url in ["/", "/static/style.css", "/static/app.js"]:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            response.close()

    def test_invalid_modes_json_and_controls(self):
        for url in ["/stream_stats/invalid", "/session_summary/invalid"]:
            self.assertEqual(self.client.get(url).status_code, 400)
        for body in [[], {}, {"mode": [], "action": "play"},
                     {"mode": "video", "action": []}, {"mode": "video", "action": "unknown"},
                     {"mode": "video", "action": "seek", "time": "bad"},
                     {"mode": "video", "action": "seek", "time": float("nan")},
                     {"mode": "video", "action": "seek", "time": True}]:
            self.assertEqual(self.client.post("/video_control", json=body).status_code, 400, body)
        self.assertEqual(self.client.post("/video_control", data="broken", content_type="application/json").status_code, 400)

    def test_upload_validation_and_size_limit(self):
        self.assertEqual(self.client.post("/upload_video").status_code, 400)
        self.assertEqual(self.upload(self.client, "payload.txt").status_code, 400)
        self.app.config["MAX_CONTENT_LENGTH"] = 10
        self.assertEqual(self.upload(self.client).status_code, 413)
        self.assertFalse(list(Path(self.directory.name).rglob("*.mp4")))

    def test_corrupt_video_rejected_with_real_decoder(self):
        app = create_app(lambda: EmptyDetector(), test_config={"TESTING": True, "UPLOAD_FOLDER": self.directory.name})
        client = app.test_client()
        response = self.upload(client)
        self.assertEqual(response.status_code, 422)
        self.assertFalse(list(Path(self.directory.name).rglob("*.mp4")))

    def test_uploads_and_analytics_are_browser_scoped(self):
        second = self.app.test_client()
        self.assertEqual(self.upload(self.client).status_code, 200)
        self.assertFalse(second.get("/check_video_uploaded").get_json()["uploaded"])
        self.assertEqual(second.get("/video_feed_file").status_code, 404)
        self.assertEqual(self.upload(second).status_code, 200)
        sessions = list(self.app.extensions["tracker_sessions"].sessions.values())
        self.assertNotEqual(sessions[0].video_path, sessions[1].video_path)
        sessions[0].streams["video"].analytics.stats["time"] = 12
        self.assertEqual(second.get("/session_summary/video").get_json()["elapsed"], 0)

    def test_stream_response_and_control_lifecycle(self):
        self.upload(self.client)
        response = self.client.get("/video_feed_file", buffered=False)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(next(iter(response.response)).startswith(b"--frame"))
        for action, expected in [("pause", 200), ("play", 200), ("stop", 200)]:
            self.assertEqual(self.client.post("/video_control", json={"mode": "video", "action": action}).status_code, expected)
        response.close()
        self.assertTrue(FakeCapture.instances[-1].released)

    def test_camera_has_one_owner_and_releases_on_disconnect(self):
        second = self.app.test_client()
        response = self.client.get("/video_feed_live", buffered=False)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(second.get("/video_feed_live").status_code, 409)
        response.close()
        replacement = second.get("/video_feed_live", buffered=False)
        self.assertEqual(replacement.status_code, 200)
        replacement.close()

    def test_sse_payload(self):
        response = self.client.get("/stream_stats/video", buffered=False)
        payload = next(iter(response.response)).decode()
        self.assertTrue(payload.startswith("data: "))
        self.assertEqual(json.loads(payload[6:])["hit_count"], 0)
        response.close()

    def test_no_model_load_for_page_or_validation(self):
        with patch("stream.get_detector", side_effect=AssertionError("Model loaded eagerly")):
            app = create_app(test_config={"TESTING": True})
            client = app.test_client()
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/video_feed_file").status_code, 404)


if __name__ == "__main__":
    unittest.main()
