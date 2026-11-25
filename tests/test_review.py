import io
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from app import create_app
from review import VideoReview
from tests.test_app import EmptyDetector, FakeCapture


class ReviewTests(unittest.TestCase):
    def test_review_reads_every_frame_from_zero_and_releases_capture(self):
        job = VideoReview(Path("test.mp4"), EmptyDetector, FakeCapture)
        job.thread.start()
        job.thread.join(3)
        snapshot = job.snapshot()
        self.assertEqual(snapshot["status"], "complete")
        self.assertEqual(snapshot["result"]["total_frames"], 3)
        self.assertEqual(snapshot["result"]["analysis_scope"], "entire_recording")
        self.assertAlmostEqual(snapshot["result"]["elapsed"], round(3 / 29.97, 3))
        self.assertTrue(FakeCapture.instances[-1].released)

    def test_cancel_releases_capture_without_partial_results(self):
        job = VideoReview(Path("test.mp4"), EmptyDetector, FakeCapture)
        job.cancel()
        job.thread.start()
        job.thread.join(3)
        self.assertEqual(job.snapshot()["status"], "cancelled")
        self.assertIsNone(job.snapshot()["result"])
        self.assertTrue(FakeCapture.instances[-1].released)

    def test_truncated_decode_is_not_reported_as_complete(self):
        class Truncated(FakeCapture):
            def read(self):
                return (False, None) if self.index >= 1 else super().read()
        job = VideoReview(Path("test.mp4"), EmptyDetector, Truncated)
        job.thread.start()
        job.thread.join(3)
        self.assertEqual(job.snapshot()["status"], "error")
        self.assertIsNone(job.snapshot()["result"])
        self.assertIn("frame 1 of 3", job.snapshot()["error"])

    def test_finish_ignores_seek_and_browser_reviews_are_isolated(self):
        with TemporaryDirectory() as directory:
            app = create_app(EmptyDetector, FakeCapture, test_config={"TESTING": True, "UPLOAD_FOLDER": directory})
            client = app.test_client()
            self.assertEqual(client.post("/review_video").status_code, 404)
            client.post("/upload_video", data={"video": (io.BytesIO(b"fake"), "test.mp4")})
            client.post("/start_stream/video")
            client.post("/video_control", json={"mode": "video", "action": "seek", "time": 0.05})
            self.assertEqual(client.post("/review_video").status_code, 200)
            browser = next(iter(app.extensions["tracker_sessions"].sessions.values()))
            browser.review.thread.join(3)
            summary = client.get("/review_video").get_json()
            self.assertEqual(summary["result"]["total_frames"], 3)
            self.assertFalse(browser.streams["video"].connected)
            job = browser.review
            client.post("/review_video")
            self.assertIs(browser.review, job)
            self.assertEqual(app.test_client().get("/review_video").status_code, 404)
            client.post("/upload_video", data={"video": (io.BytesIO(b"new"), "new.mp4")})
            self.assertEqual(client.get("/review_video").status_code, 404)

    def test_running_review_keeps_session_active_and_repeated_starts_reuse_job(self):
        ready, release = Event(), Event()
        def detector():
            ready.set()
            release.wait(3)
            return EmptyDetector()
        with TemporaryDirectory() as directory:
            app = create_app(detector, FakeCapture, test_config={"TESTING": True, "UPLOAD_FOLDER": directory})
            client = app.test_client()
            client.post("/upload_video", data={"video": (io.BytesIO(b"fake"), "test.mp4")})
            try:
                client.post("/review_video")
                self.assertTrue(ready.wait(3))
                browser = next(iter(app.extensions["tracker_sessions"].sessions.values()))
                self.assertTrue(browser.active)
                job = browser.review
                client.post("/review_video")
                self.assertIs(browser.review, job)
                client.post("/cancel_review")
            finally:
                release.set()
                browser.review.thread.join(3)
            self.assertEqual(browser.review.snapshot()["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
