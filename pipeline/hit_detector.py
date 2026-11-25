"""Infer impacts from measured falling-to-rising reversals, with noise gates."""
from dataclasses import asdict, dataclass


@dataclass
class HitEvent:
    timestamp: float
    streak_position: int
    y_at_impact: float
    velocity_at_impact: float
    hand_check: str = "not_checked"
    type: str = "HIT"

    def to_dict(self):
        return asdict(self)


@dataclass
class MissEvent:
    timestamp: float
    reason: str
    streak_before_miss: int
    type: str = "MISS"

    def to_dict(self):
        return asdict(self)


@dataclass
class RejectedHit:
    timestamp: float
    reason: str
    y_at_impact: float
    velocity_at_impact: float
    contact_sample_timestamp: float
    type: str = "REJECTED"

    def to_dict(self):
        return asdict(self)


class HitDetector:
    def __init__(self, cooldown_frames=6, miss_timeout_seconds=3.0,
                 velocity_threshold=0.01, min_fall_velocity=0.05,
                 min_impact_acceleration=0.0, min_fall_distance=0.0, min_fall_duration=0.0):
        if min(cooldown_frames, velocity_threshold, min_fall_velocity, min_impact_acceleration, min_fall_distance, min_fall_duration) < 0:
            raise ValueError("Detection thresholds cannot be negative")
        if miss_timeout_seconds <= 0:
            raise ValueError("Miss timeout must be positive")
        self.cooldown_frames = cooldown_frames
        self.miss_timeout_seconds = miss_timeout_seconds
        self.velocity_threshold = velocity_threshold
        self.min_fall_velocity = min_fall_velocity
        self.min_impact_acceleration = min_impact_acceleration
        self.min_fall_distance = min_fall_distance
        self.min_fall_duration = min_fall_duration
        self.reset()

    def reset(self):
        self.events = []
        self.rejections = []
        self._current_streak = 0
        self._streak_history = []
        self._last_hit_time = None
        self._cooldown_until = float("-inf")
        self._finalized = False
        self.reset_motion()

    def reset_motion(self):
        """Never infer a reversal across an unobserved interval."""
        self._falling = False
        self._peak_fall_vel = 0.0
        self._peak_accel = 0.0
        self._fall_start = None
        self._fall_min_y = self._fall_max_y = 0.0

    def process(self, y_smooth, velocity, fps, timestamp, acceleration=0.0, candidate_validator=None):
        if self._finalized:
            return None
        result = self._check_timeout(timestamp, "timeout")
        if velocity > self.velocity_threshold:
            if not self._falling:
                self._fall_start = timestamp
                self._fall_min_y = self._fall_max_y = y_smooth
            self._falling = True
            self._peak_fall_vel = max(self._peak_fall_vel, velocity)
        if self._falling:
            if velocity >= -self.velocity_threshold:
                self._fall_max_y = max(self._fall_max_y, y_smooth)
            self._peak_accel = max(self._peak_accel, abs(acceleration))
        if self._falling and velocity < -self.velocity_threshold:
            if (timestamp >= self._cooldown_until
                    and self._fall_max_y - self._fall_min_y >= self.min_fall_distance
                    and timestamp - self._fall_start >= self.min_fall_duration
                    and self._peak_fall_vel >= self.min_fall_velocity
                    and self._peak_accel >= self.min_impact_acceleration):
                decision = candidate_validator() if candidate_validator is not None else None
                if decision is not None and not decision.allowed:
                    result = RejectedHit(round(timestamp, 6), decision.reason, round(y_smooth, 6),
                                         round(velocity, 6), round(decision.sample_timestamp, 6))
                    self.rejections.append(result)
                else:
                    self._current_streak += 1
                    result = HitEvent(round(timestamp, 6), self._current_streak,
                                      round(y_smooth, 6), round(velocity, 6),
                                      hand_check=decision.reason if decision is not None else "not_checked")
                    self.events.append(result)
                    self._last_hit_time = timestamp
                    self._cooldown_until = timestamp + self.cooldown_frames / max(fps, 1)
            self.reset_motion()
        return result

    def process_lost_frame(self, fps, timestamp):
        self.reset_motion()
        return None if self._finalized else self._check_timeout(timestamp, "ball_lost")

    def _check_timeout(self, timestamp, reason):
        if self._last_hit_time is None or timestamp - self._last_hit_time < self.miss_timeout_seconds:
            return None
        event = MissEvent(round(timestamp, 6), reason, self._current_streak)
        self.events.append(event)
        self._close_streak()
        self.reset_motion()
        return event

    def _close_streak(self):
        if self._current_streak:
            self._streak_history.append(self._current_streak)
        self._current_streak = 0
        self._last_hit_time = None

    def finalize_session(self, timestamp=None):
        """Stopping or reaching EOF closes the streak without inventing a miss."""
        if not self._finalized:
            self._close_streak()
            self.reset_motion()
            self._finalized = True

    def get_current_streak(self):
        return self._current_streak

    def get_total_hits(self):
        return sum(event.type == "HIT" for event in self.events)

    def get_total_misses(self):
        return sum(event.type == "MISS" for event in self.events)

    def get_streak_history(self):
        return list(self._streak_history)

    def get_timeline(self):
        return [event.to_dict() for event in self.events]

    def get_timeline_text(self):
        segments, run = [], 0
        for event in self.events:
            if event.type == "HIT":
                run += 1
            else:
                if run:
                    segments.append(f"{run} hit{'s' if run != 1 else ''}")
                    run = 0
                segments.append("MISS")
        if run:
            segments.append(f"{run} hit{'s' if run != 1 else ''}")
        return " -> ".join(segments) or "No events yet"

    def get_summary(self):
        return {
            "total_hits": self.get_total_hits(), "total_misses": self.get_total_misses(),
            "current_streak": self._current_streak, "streak_history": self.get_streak_history(),
            "best_streak": max([self._current_streak, *self._streak_history]),
            "timeline": self.get_timeline(), "timeline_text": self.get_timeline_text(),
            "rejected_count": len(self.rejections),
            "hand_rejected_count": sum(event.reason == "hand_contact" for event in self.rejections),
            "unverified_count": sum(event.reason != "hand_contact" for event in self.rejections),
            "uncertain_hit_count": sum(isinstance(event, HitEvent) and event.hand_check == "hand_uncertain" for event in self.events),
            "rejections": [event.to_dict() for event in self.rejections],
        }
