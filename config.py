import os

# ── Application Configuration ─────────────────────────────────────
UPLOAD_FOLDER = "outputs"
CUSTOM_VIDEO_PATH = os.path.join(UPLOAD_FOLDER, "custom_video.mp4")

# ── YOLO Model Settings ───────────────────────────────────────────
MODEL_PATH = "yolov8n-f2.pt"
CONFIDENCE_THRESHOLD = 0.2
TARGET_CLASS = "golf_ball"

# ── Metrics & Conversions ─────────────────────────────────────────
# Approximate pixels-per-meter for your camera setup.
# Calibrate by measuring a known real-world distance and counting the pixel span.
PIXELS_PER_METER = 2650.0

# ── Signal Processing (Savitzky-Golay) ────────────────────────────
SG_BUFFER_SIZE = 7
SG_POLY_ORDER  = 2
HISTORY_CAP    = 5000  # Max stored velocity/accel samples per session

# ── Flask Server ──────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True
THREADED = True
