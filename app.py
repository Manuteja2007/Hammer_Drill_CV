"""Flask entry point for browser-scoped hammer-drill analysis."""
import math
import os
from pathlib import Path
from secrets import token_hex
from uuid import uuid4

import cv2
from flask import Flask, Response, g, jsonify, render_template, request, session, url_for
from werkzeug.exceptions import RequestEntityTooLarge

import config
from review import VideoReview
from state import SessionRegistry
from stream import StreamController, StreamError, get_detector, stream_stats_generator


def create_app(detector_factory=None, capture_factory=None, registry=None, test_config=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("TRACKER_SECRET_KEY") or token_hex(32),
        MAX_CONTENT_LENGTH=config.MAX_UPLOAD_BYTES,
        UPLOAD_FOLDER=str(config.UPLOAD_FOLDER),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    if test_config:
        app.config.update(test_config)
    registry = registry if registry is not None else SessionRegistry()
    app.extensions["tracker_sessions"] = registry
    capture_factory = capture_factory or cv2.VideoCapture
    detector_factory = detector_factory or get_detector

    @app.before_request
    def identify_browser():
        if "tracker_id" not in session:
            session["tracker_id"] = uuid4().hex
        try:
            g.browser = registry.get(session["tracker_id"])
        except RuntimeError as exc:
            raise StreamError(str(exc), 503) from exc

    def controller(mode):
        if not isinstance(mode, str) or mode not in {"live", "video"}:
            raise StreamError("Invalid mode.")
        with g.browser.lock:
            if mode not in g.browser.streams:
                g.browser.streams[mode] = StreamController(mode, detector_factory, capture_factory)
            return g.browser.streams[mode]

    @app.errorhandler(StreamError)
    def stream_error(exc):
        return jsonify(error=str(exc)), exc.status

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(exc):
        return jsonify(error="Video exceeds the 500 MB upload limit."), 413

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.after_request
    def fresh_interface(response):
        response.headers["Cache-Control"] = "no-cache" if request.endpoint == "static" else "no-store"
        return response

    @app.post("/upload_video")
    def upload_video():
        file = request.files.get("video")
        if file is None or not file.filename:
            raise StreamError("Select a video file.")
        suffix = Path(file.filename).suffix.lower()
        if suffix not in config.ALLOWED_VIDEO_EXTENSIONS:
            raise StreamError("Supported formats: MP4, AVI, MOV, MKV, WEBM.")
        directory = Path(app.config["UPLOAD_FOLDER"]) / session["tracker_id"]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (uuid4().hex + suffix)
        try:
            file.save(path)
            probe = capture_factory(str(path))
            try:
                readable, frame = probe.read() if probe.isOpened() else (False, None)
                if not readable or frame is None:
                    raise StreamError("The uploaded file is not a readable video.", 422)
            finally:
                probe.release()
        except Exception:
            path.unlink(missing_ok=True)
            raise
        with g.browser.lock:
            if g.browser.review is not None:
                g.browser.review.cancel()
                g.browser.review = None
            controller("video").stop()
            g.browser.video_path = path
        return jsonify(message="Upload successful")

    @app.post("/video_control")
    def video_control():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise StreamError("Provide a JSON object.")
        action = data.get("action")
        if not isinstance(action, str) or action not in {"play", "pause", "seek", "stop"}:
            raise StreamError("Invalid playback action.")
        position = None
        if action == "seek":
            try:
                if isinstance(data.get("time"), bool):
                    raise ValueError
                position = float(data["time"])
                if not math.isfinite(position):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                raise StreamError("Seek time must be a finite number.") from None
        stream = controller(data.get("mode"))
        stream.control(action, position)
        return jsonify(status="success")

    @app.get("/check_video_uploaded")
    def check_video():
        path = g.browser.video_path
        return jsonify(uploaded=bool(path and path.is_file()))

    @app.post("/start_stream/<mode>")
    def start_stream(mode):
        stream = controller(mode)
        with g.browser.lock:
            source = 0 if mode == "live" else g.browser.video_path
            if mode == "video" and (source is None or not source.is_file()):
                raise StreamError("Upload a video first.", 404)
            token = stream.open(0 if mode == "live" else str(source))
        endpoint = "video_feed_live" if mode == "live" else "video_feed_file"
        return jsonify(feed_url=url_for(endpoint, token=token))

    def video_response(mode, source):
        stream = controller(mode)
        token = request.args.get("token")
        if token is None:
            token = stream.open(source)
        stream.attach(token)
        response = Response(stream.generate(token), mimetype="multipart/x-mixed-replace; boundary=frame",
                            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        # Also close captures if the response is closed before its generator starts.
        def close_stream():
            with stream.lock:
                if stream.token == token:
                    stream.stop()
        response.call_on_close(close_stream)
        return response

    @app.get("/video_feed_live")
    def video_feed_live():
        return video_response("live", 0)

    @app.get("/video_feed_file")
    def video_feed_file():
        with g.browser.lock:
            path = g.browser.video_path
            if path is None or not path.is_file():
                raise StreamError("Upload a video first.", 404)
            return video_response("video", str(path))

    @app.get("/stream_stats/<mode>")
    def stream_stats(mode):
        return Response(stream_stats_generator(controller(mode), g.browser), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.post("/review_video")
    def review_video():
        with g.browser.lock:
            path = g.browser.video_path
            if path is None or not path.is_file():
                raise StreamError("Upload a video first.", 404)
            controller("video").stop()
            job = g.browser.review
            if job is None or job.snapshot()["status"] in {"error", "cancelled"}:
                job = VideoReview(path, detector_factory, capture_factory)
                g.browser.review = job
                job.thread.start()
            return jsonify(job.snapshot())

    @app.get("/review_video")
    def review_progress():
        with g.browser.lock:
            if g.browser.review is None:
                raise StreamError("Start a full recording review first.", 404)
            return jsonify(g.browser.review.snapshot())

    @app.post("/cancel_review")
    def cancel_review():
        with g.browser.lock:
            if g.browser.review is not None:
                g.browser.review.cancel()
        return jsonify(status="cancelled")

    @app.get("/session_summary/<mode>")
    def session_summary(mode):
        stream = controller(mode)
        with g.browser.lock:
            if mode == "video" and g.browser.review is not None:
                job = g.browser.review.snapshot()
                if job["status"] == "complete":
                    return jsonify(job["result"])
        return jsonify(stream.summary())

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
