"""Browser-scoped state and constant-memory motion aggregates."""
from dataclasses import dataclass, field
from threading import RLock
import time

import config


def default_stats():
    return {
        "v_y": "--", "y_smooth": "--", "accel_y": "--", "state": "Waiting",
        "time": 0.0, "duration": 0.0, "fps": 0.0,
        "ball_x": "--", "ball_y": "--", "confidence": "--", "direction": "--",
        "max_v_y": 0.0, "avg_v_y": 0.0, "hit_count": 0, "miss_count": 0,
        "current_streak": 0, "best_streak": 0, "last_event": "--",
        "timeline_text": "No events yet", "play_state": "pause",
        "hand_status": "Checks run at candidate taps", "hand_check_ms": 0.0,
        "rejected_count": 0, "hand_rejected_count": 0, "unverified_count": 0,
        "uncertain_hit_count": 0,
    }


class AnalyticsState:
    def __init__(self):
        self.reset()

    def reset(self, duration=0.0, fps=0.0, position=0.0):
        self.stats = default_stats()
        self.stats.update(duration=duration, fps=fps, time=position)
        self.total_frames = self.detected_frames = self.lost_frames = 0
        self.sample_count = 0
        self.velocity_sum = self.acceleration_sum = 0.0
        self.hand_check_count = 0
        self.hand_check_total_ms = 0.0
        self.hit_summary = {
            "total_hits": 0, "total_misses": 0, "current_streak": 0,
            "best_streak": 0, "streak_history": [], "timeline": [],
            "timeline_text": "No events yet",
            "rejected_count": 0, "hand_rejected_count": 0, "unverified_count": 0, "rejections": [],
            "uncertain_hit_count": 0,
        }

    def record_motion(self, result):
        self.sample_count += 1
        self.velocity_sum += result.velocity
        self.acceleration_sum += result.acceleration
        self.stats.update(
            y_smooth=round(result.y_smooth, 6), v_y=round(result.velocity, 6),
            accel_y=round(result.acceleration, 6), direction=result.direction,
            max_v_y=max(self.stats["max_v_y"], abs(result.velocity)),
            avg_v_y=self.velocity_sum / self.sample_count,
        )

    def record_hand_check(self, decision):
        if decision.reason == "disabled":
            self.stats.update(hand_status="Disabled", hand_check_ms=0.0)
            return
        self.hand_check_count += 1
        self.hand_check_total_ms += decision.latency_ms
        self.stats.update(hand_status=decision.status, hand_check_ms=round(decision.latency_ms, 2))

    def sync_hits(self, detector):
        self.hit_summary = detector.get_summary()
        summary = self.hit_summary
        self.stats.update(hit_count=summary["total_hits"], miss_count=summary["total_misses"],
                          current_streak=summary["current_streak"], best_streak=summary["best_streak"],
                          timeline_text=summary["timeline_text"], rejected_count=summary["rejected_count"],
                          hand_rejected_count=summary["hand_rejected_count"], unverified_count=summary["unverified_count"],
                          uncertain_hit_count=summary["uncertain_hit_count"])
        audit = summary["timeline"] + summary["rejections"]
        if audit:
            self.stats["last_event"] = sorted(audit, key=lambda event: event["timestamp"])[-1]["type"]

    def summary(self):
        return {
            **self.hit_summary,
            "streaks": list(self.hit_summary["streak_history"]),
            "elapsed": round(self.stats["time"], 3),
            "max_velocity": round(self.stats["max_v_y"], 6),
            "avg_velocity": round(self.velocity_sum / self.sample_count, 6) if self.sample_count else 0,
            "avg_accel": round(self.acceleration_sum / self.sample_count, 6) if self.sample_count else 0,
            "total_frames": self.total_frames, "detected_frames": self.detected_frames,
            "lost_frames": self.lost_frames,
            "hand_check_count": self.hand_check_count,
            "avg_hand_check_ms": round(self.hand_check_total_ms / self.hand_check_count, 3) if self.hand_check_count else 0.0,
        }


@dataclass
class BrowserSession:
    streams: dict = field(default_factory=dict)
    video_path: object = None
    review: object = None
    last_seen: float = field(default_factory=time.monotonic)
    lock: object = field(default_factory=RLock)

    @property
    def active(self):
        with self.lock:
            if self.review is not None and self.review.snapshot()["status"] == "running":
                return True
            for stream in self.streams.values():
                # A model load or frame may be in progress before connected is set.
                if not stream.lock.acquire(blocking=False):
                    return True
                try:
                    stream.expire_prepared()
                    if stream.connected:
                        return True
                finally:
                    stream.lock.release()
            return False

    def close(self):
        with self.lock:
            if self.review is not None:
                self.review.cancel()
            for stream in self.streams.values():
                stream.stop()


class SessionRegistry:
    """Bound idle memory usage; active streams are never evicted."""
    def __init__(self):
        self.sessions = {}
        self.lock = RLock()

    def get(self, session_id):
        with self.lock:
            now = time.monotonic()
            expired = [key for key, value in self.sessions.items()
                       if not value.active and now - value.last_seen > config.SESSION_TTL_SECONDS]
            for key in expired:
                self.sessions.pop(key).close()
            if session_id not in self.sessions:
                if len(self.sessions) >= config.MAX_SESSIONS:
                    idle = [(value.last_seen, key) for key, value in self.sessions.items() if not value.active]
                    if not idle:
                        raise RuntimeError("All session slots are busy. Please try again later.")
                    self.sessions.pop(min(idle)[1]).close()
                self.sessions[session_id] = BrowserSession()
            result = self.sessions[session_id]
            result.last_seen = now
            return result
