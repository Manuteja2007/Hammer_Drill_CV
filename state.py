import time

class SessionState:
    """Encapsulates all mutable global state for the application."""
    
    def __init__(self):
        self.stream_controls = {
            "live":  {"state": "pause", "seek": None},
            "video": {"state": "pause", "seek": None, "duration": 0},
        }
        
        self.session_analytics = {"live": {}, "video": {}}
        
        self.latest_stats = {
            "live": self.default_stats(),
            "video": self.default_stats(include_duration=True),
        }
        
        self.video_uploaded = False
        
        self.reset_analytics("live")
        self.reset_analytics("video")

    def default_stats(self, include_duration=False):
        """Return a fresh stats dictionary for SSE with placeholder values."""
        stats = {
            "v_y": "--",  "y_smooth": "--",  "accel_y": "--",
            "state": "Waiting",  "time": 0,
            "ball_x": "--",  "ball_y": "--",  "confidence": "--",
            "direction": "--",  "fps": 0,
            "max_v_y": "--",  "avg_v_y": "--",
        }
        if include_duration:
            stats["duration"] = 0
        return stats

    def default_analytics(self):
        """Return a fresh session-analytics bucket."""
        return {
            "total_frames": 0,
            "detected_frames": 0,
            "lost_frames": 0,
            "all_v_y": [],
            "all_accel_y": [],
            "all_y_smooth": [],
            "max_v_y": 0.0,
            "min_v_y": 0.0,
            "start_time": time.time(),
            "elapsed": 0.0,       # only ticks while playing (live)
            "last_tick": None,
        }

    def reset_analytics(self, mode):
        """Clear accumulated analytics for the given mode."""
        self.session_analytics[mode] = self.default_analytics()

# Create a singleton instance to be used across the app
global_state = SessionState()
