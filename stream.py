"""Managed MJPEG streams: isolated controls, synchronized lifecycle, and pacing."""
import json
import math
import time
from threading import Event, Lock, RLock
from uuid import uuid4

import cv2

import config
from pipeline import Annotator, YOLODetector
from pipeline.processor import FrameProcessor
from state import AnalyticsState

_model_lock = Lock()
_camera_lock = Lock()
_detector = None


class StreamError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def get_detector():
    global _detector
    with _model_lock:
        if _detector is None:
            try:
                _detector = YOLODetector(config.MODEL_PATH, config.CONFIDENCE_THRESHOLD, config.TARGET_CLASS)
            except Exception as exc:
                raise StreamError("The detection model could not be loaded. Check the model file.", 503) from exc
        return _detector


class StreamController:
    def __init__(self, mode, detector_factory=get_detector, capture_factory=cv2.VideoCapture):
        self.mode = mode
        self.detector_factory = detector_factory
        self.capture_factory = capture_factory
        self.lock = RLock()
        self.wake = Event()
        self.analytics = AnalyticsState()
        self.processor = None
        self.cap = None
        self.source = None
        self.connected = False
        self.attached = False
        self.prepared_at = None
        self.camera_owned = False
        self.ended = False
        self.token = None
        self.last_jpeg = None
        self.frame_index = 0
        self.anchor_wall = None
        self.anchor_media = 0.0

    def _open_capture(self):
        if self.mode == "live" and not self.camera_owned:
            if not _camera_lock.acquire(blocking=False):
                raise StreamError("The camera is already in use by another session.", 409)
            self.camera_owned = True
        try:
            self.cap = self.capture_factory(self.source)
            if not self.cap.isOpened():
                raise StreamError("Cannot open the camera or video file.", 422)
        except Exception:
            self._release_capture()
            raise

    def _release_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.camera_owned:
            _camera_lock.release()
            self.camera_owned = False

    def open(self, source):
        with self.lock:
            if self.connected:
                raise StreamError("This session already has an open stream. Stop it before restarting.", 409)
            detector = self.detector_factory()
            self.source = source
            self._open_capture()
            try:
                reported_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
                self.fps = reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else 30.0
                count = float(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.duration = max(0.0, count / self.fps) if self.mode == "video" and math.isfinite(count) else 0.0
                self.processor = FrameProcessor(detector, self.analytics)
                self.processor.reset(duration=self.duration, fps=self.fps)
                self.analytics.stats["play_state"] = "play"
                self.connected, self.ended = True, False
                self.attached = False
                self.prepared_at = time.monotonic()
                self.token = uuid4().hex
                self.frame_index = 0
                self.last_jpeg = None
                self.anchor_wall = None
                return self.token
            except Exception:
                self._release_capture()
                raise

    def attach(self, token):
        """Only one MJPEG consumer may advance a session's capture."""
        with self.lock:
            self.expire_prepared()
            if not self.connected or self.token != token:
                raise StreamError("This stream is no longer available. Start a new session.", 409)
            if self.attached:
                raise StreamError("This stream is already open in another view.", 409)
            self.attached = True

    def expire_prepared(self):
        """Release a preflight capture if the browser never attaches its image."""
        with self.lock:
            if (self.connected and not self.attached and self.prepared_at is not None
                    and time.monotonic() - self.prepared_at >= config.STREAM_START_TIMEOUT_SECONDS):
                self.stop()

    def control(self, action, position=None):
        with self.lock:
            if action == "stop":
                self.stop()
                return
            if not self.connected:
                raise StreamError("Start a stream before changing playback.", 409)
            if action == "seek":
                if self.mode != "video":
                    raise StreamError("Seeking is only supported for uploaded video.")
                if position is None or not math.isfinite(position) or not 0 <= position <= self.duration:
                    raise StreamError("Seek time must be within the video duration.")
                self._seek(position)
            elif action == "play":
                if self.ended:
                    if self.mode == "video":
                        self._seek(0.0)
                    else:
                        self._open_capture()
                        self.processor.reset(fps=self.fps)
                        self.frame_index = 0
                        self.last_jpeg = None
                        self.ended = False
                self.analytics.stats["play_state"] = "play"
            elif action == "pause":
                self.analytics.stats["play_state"] = "pause"
                # Do not infer a reversal between measurements separated by a pause.
                self.processor.filter.reset()
                self.processor.tracker.reset()
                self.processor.last_measurement_time = None
                self.processor.hit_detector.reset_motion()
                self.processor.hand_guard.reset()
                self.processor.last_hand_overlay = None
            else:
                raise StreamError("Invalid playback action.")
            self.anchor_wall = None
            self.wake.set()

    def _seek(self, position):
        if self.cap is None:
            self._open_capture()
        index = int(position * self.fps)
        if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise StreamError("This video does not support seeking.", 422)
        play_state = self.analytics.stats["play_state"]
        self.frame_index = index
        self.processor.reset(self.duration, self.fps, index / self.fps)
        self.analytics.stats["play_state"] = play_state
        self.ended = False
        self.last_jpeg = None

    def _finish(self, label):
        if self.processor is not None:
            self.processor.finalize()
        self.analytics.stats.update(play_state="pause", state=label)
        self._release_capture()
        self.ended = True
        self.anchor_wall = None

    def stop(self):
        with self.lock:
            self._finish(self.analytics.stats["state"] if self.ended else "Session Complete")
            self.connected = False
            self.attached = False
            self.prepared_at = None
            self.token = None
            self.wake.set()

    def snapshot(self):
        with self.lock:
            return dict(self.analytics.stats)

    def summary(self):
        with self.lock:
            return self.analytics.summary()

    def generate(self, token):
        try:
            while True:
                delay, jpeg = 0.1, None
                with self.lock:
                    if not self.connected or self.token != token:
                        return
                    self.wake.clear()
                    if self.analytics.stats["play_state"] == "pause":
                        jpeg = self.last_jpeg
                    else:
                        now = time.monotonic()
                        if self.anchor_wall is None:
                            self.anchor_wall = now
                            self.anchor_media = self.analytics.stats["time"]
                        ok, frame = self.cap.read()
                        if not ok:
                            if self.mode == "video":
                                self.analytics.stats["time"] = self.frame_index / self.fps
                                self._finish("Video Ended")
                            else:
                                self._finish("Camera Disconnected")
                            jpeg = self.last_jpeg
                        else:
                            self.frame_index += 1
                            if self.mode == "video":
                                timestamp = self.frame_index / self.fps
                            else:
                                timestamp = self.anchor_media + time.monotonic() - self.anchor_wall
                                frame = cv2.flip(frame, 1)
                            self.processor.process(frame, self.fps, timestamp)
                            jpeg = Annotator.encode_jpeg(frame)
                            self.last_jpeg = jpeg
                            delay = (max(0.0, self.anchor_wall + timestamp - self.anchor_media - time.monotonic())
                                     if self.mode == "video" else 0.0)
                # Never hold the control lock while waiting or sending a frame.
                if delay:
                    self.wake.wait(delay)
                with self.lock:
                    if not self.connected or self.token != token:
                        return
                if jpeg is not None:
                    yield Annotator.format_multipart_jpeg(jpeg)
        except Exception:
            with self.lock:
                self._finish("Processing Error")
            raise
        finally:
            with self.lock:
                if self.token == token:
                    self.stop()


def stream_stats_generator(controller, browser_session):
    while True:
        browser_session.last_seen = time.monotonic()
        yield f"data: {json.dumps(controller.snapshot())}\n\n"
        time.sleep(0.1)
