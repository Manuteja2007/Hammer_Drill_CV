"""Application settings. Paths are anchored to the project, not the shell."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "yolov8n-f2.pt"
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"
DEFAULT_VIDEO_PATH = BASE_DIR / "data" / "videos" / "Javed.mp4"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
CONFIDENCE_THRESHOLD = 0.2
TARGET_CLASS = "golf_ball"
# Calibrate for the recording resolution and camera distance.
PIXELS_PER_METER = 2650.0
SG_BUFFER_SIZE = 7
SG_POLY_ORDER = 2
MAX_MEASUREMENT_GAP_SECONDS = 0.5
HIT_COOLDOWN_FRAMES = 6
MISS_TIMEOUT_SECONDS = 3.0
# Require a fall above +threshold, followed by a rise below -threshold.
HIT_VELOCITY_THRESHOLD = 0.01
HIT_MIN_FALL_VELOCITY = 0.05
# Reject short, small reversals; tune to the calibrated camera scale.
HIT_MIN_FALL_DISTANCE = 0.006  # Approx. 16 pixels at the preset scale.
HIT_MIN_FALL_DURATION = 0.04  # Seconds of observed motion before reversal.
HIT_MIN_IMPACT_ACCELERATION = 0.0  # Zero disables this optional heuristic.
# MediaPipe hand models run locally through the already-required OpenCV package.
HAND_CHECK_ENABLED = True
HAND_PALM_MODEL_PATH = BASE_DIR / "models" / "hands" / "palm_detection_mediapipe_2023feb.onnx"
HAND_LANDMARK_MODEL_PATH = BASE_DIR / "models" / "hands" / "handpose_estimation_mediapipe_2023feb.onnx"
HAND_PALM_CONFIDENCE = 0.55
HAND_LANDMARK_CONFIDENCE = 0.60
HAND_MAX_NEARBY_HANDS = 2
HAND_PALM_REACH_FACTOR = 3.0
HAND_ROI_MIN_SIDE = 320
HAND_ROI_BALL_DIAMETERS = 20
HAND_CONTACT_MARGIN_PIXELS = 2.0
HAND_FINGER_RADIUS_PALM_RATIO = 0.10
HAND_CONTACT_LOOKBACK_SECONDS = 0.12
HAND_HISTORY_FRAMES = 16
# A weak pose is not evidence of contact. True is available for stricter operation.
HAND_REJECT_UNCERTAIN = False
HOST = os.environ.get("TRACKER_HOST", "127.0.0.1")
PORT = int(os.environ.get("TRACKER_PORT", "5000"))
DEBUG = os.environ.get("TRACKER_DEBUG", "0") == "1"
SESSION_TTL_SECONDS = 24 * 60 * 60
MAX_SESSIONS = 128
STREAM_START_TIMEOUT_SECONDS = 30
