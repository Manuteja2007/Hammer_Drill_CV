"""Full-recording analysis, independent of the preview playback position."""
from threading import Event, RLock, Thread
import math

import cv2

from pipeline.processor import FrameProcessor


class VideoReview:
    def __init__(self, path, detector_factory, capture_factory):
        self.path = path
        self.lock = RLock()
        self.cancelled = Event()
        self.status = "running"
        self.processed = self.total = 0
        self.result = self.error = None
        self.thread = Thread(target=self._run, args=(detector_factory, capture_factory), daemon=True)

    def snapshot(self):
        with self.lock:
            return dict(status=self.status, processed_frames=self.processed, total_frames=self.total,
                        result=self.result, error=self.error)

    def cancel(self):
        self.cancelled.set()

    def _run(self, detector_factory, capture_factory):
        cap = None
        try:
            cap = capture_factory(str(self.path))
            if not cap.isOpened():
                raise ValueError("Unable to open this recording for review.")
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if math.isfinite(fps) and fps > 0 else 30.0
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            with self.lock:
                self.total = max(0, int(count)) if math.isfinite(count) else 0
            processor = FrameProcessor(detector_factory())
            processor.reset(duration=self.total / fps, fps=fps)
            while not self.cancelled.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                processor.process(frame, fps, (self.processed + 1) / fps, annotate=False)
                with self.lock:
                    self.processed += 1
            if not self.cancelled.is_set():
                if not self.processed:
                    raise ValueError("The recording contains no readable frames.")
                # Do not silently report a truncated decode as a complete review.
                if self.total and self.processed < self.total:
                    raise ValueError(f"Video decoding stopped at frame {self.processed} of {self.total}.")
                processor.analytics.stats["time"] = self.processed / fps
                processor.finalize()
                with self.lock:
                    self.result = {**processor.analytics.summary(), "analysis_scope": "entire_recording"}
                    self.status = "complete"
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.status = "error"
        finally:
            if cap is not None:
                cap.release()
            if self.cancelled.is_set():
                with self.lock:
                    self.status = "cancelled"
                    self.result = None
