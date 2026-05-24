"""
YOLO Vision Tracker — Flask Entry Point
====================================
Modular, pipeline-based backend for real-time video analytics.

Endpoints:
    GET  /                      → SPA (index.html)
    GET  /video_feed_live       → MJPEG live webcam stream
    GET  /video_feed_file       → MJPEG uploaded-video stream
    GET  /stream_stats/<mode>   → SSE real-time analytics
    POST /upload_video          → Upload a video file
    POST /video_control         → Play / Pause / Seek
    GET  /session_summary/<mode>→ Final session report
"""

import os
import numpy as np
from flask import Flask, Response, render_template, request, jsonify

import config
from state import global_state
from stream import generate_frames, stream_stats_generator

# ── App Setup ──────────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)


# ── Routes ─────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the single-page application."""
    return render_template("index.html")

@app.route("/upload_video", methods=["POST"])
def upload_video():
    """Accept and save a user-uploaded video file."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file.save(config.CUSTOM_VIDEO_PATH)
    global_state.video_uploaded = True
    global_state.stream_controls["video"]["state"] = "play"
    return jsonify({"message": "Upload successful"}), 200

@app.route("/video_control", methods=["POST"])
def video_control():
    """Handle play / pause / seek commands from the frontend."""
    data   = request.json
    mode   = data.get("mode")
    action = data.get("action")

    if mode not in global_state.stream_controls:
        return jsonify({"error": "Invalid mode"}), 400

    if action in ("pause", "play"):
        global_state.stream_controls[mode]["state"] = action
    elif action == "seek":
        global_state.stream_controls[mode]["seek"] = float(data.get("time", 0))

    return jsonify({"status": "success"}), 200

@app.route("/check_video_uploaded")
def check_video():
    """Quick check whether a video has been uploaded."""
    return jsonify({"uploaded": global_state.video_uploaded})

@app.route("/video_feed_live")
def video_feed_live():
    """MJPEG stream from the webcam."""
    global_state.stream_controls["live"]["state"] = "play"
    return Response(
        generate_frames(0, "live"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/video_feed_file")
def video_feed_file():
    """MJPEG stream from the uploaded video."""
    path = config.CUSTOM_VIDEO_PATH
    if not os.path.exists(path):
        return jsonify({"error": "No video uploaded yet"}), 404
    return Response(
        generate_frames(path, "video"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/stream_stats/<mode>")
def stream_stats(mode):
    """Server-Sent Events endpoint — pushes analytics."""
    return Response(stream_stats_generator(mode), mimetype="text/event-stream")

@app.route("/session_summary/<mode>")
def session_summary(mode):
    """Return accumulated session analytics for the results page."""
    sa = global_state.session_analytics.get(mode, {})

    elapsed = sa.get("elapsed", 0)
    if mode == "video":
        elapsed = global_state.latest_stats.get("video", {}).get("time", 0)

    avg_v = float(np.mean(sa["all_v_y"]))     if sa.get("all_v_y")     else 0
    avg_a = float(np.mean(sa["all_accel_y"])) if sa.get("all_accel_y") else 0
    max_v = sa.get("max_v_y", 0)

    return jsonify({
        "elapsed":      round(elapsed, 2),
        "max_velocity": round(max_v, 4),
        "avg_velocity": round(avg_v, 4),
        "avg_accel":    round(avg_a, 4),
    })

# ── Entry Point ────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(
        host=config.HOST, 
        port=config.PORT, 
        debug=config.DEBUG, 
        threaded=config.THREADED
    )
