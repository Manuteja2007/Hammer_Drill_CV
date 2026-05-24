"""
Stream Generator — MJPEG + SSE
================================
Composes Detector → Tracker → Filter → Annotator pipeline.
Handles pause/play/seek and real-time video pacing.
"""

import time
import json
import numpy as np
import cv2

import config
from state import global_state
from pipeline import YOLODetector, KalmanTracker, SavgolFilter, Annotator

# ── Singleton detector instance ────────────────────────────────
detector = YOLODetector(
    config.MODEL_PATH,
    config.CONFIDENCE_THRESHOLD,
    config.TARGET_CLASS,
)


def generate_frames(video_source, mode):
    """
    Core MJPEG generator.

    Reads frames from *video_source* (0 = webcam, path = file),
    runs the full detection pipeline, and yields multipart JPEG frames.
    Respects stream_controls for pause / play / seek.
    """
    if detector.model is None:
        print("[ERROR] YOLO model not loaded.")
        return

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {video_source}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # Store duration for video mode
    if mode == "video":
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = n_frames / fps if fps else 0
        global_state.stream_controls["video"]["duration"] = duration
        global_state.latest_stats["video"]["duration"] = duration

    global_state.reset_analytics(mode)
    global_state.latest_stats[mode]["fps"] = fps

    tracker   = KalmanTracker()
    sg_filter = SavgolFilter(
        config.SG_BUFFER_SIZE, config.SG_POLY_ORDER, config.PIXELS_PER_METER
    )

    last_jpeg      = None    # cached frame for pause-hold
    playback_start = None    # wall-clock time when video playback began
    video_start_ms = 0       # video position (ms) when playback began

    # ── Main loop ──────────────────────────────────────────────
    while True:
        ctrl  = global_state.stream_controls[mode]
        sa    = global_state.session_analytics[mode]
        stats = global_state.latest_stats[mode]

        # 1. Seek
        if ctrl.get("seek") is not None:
            cap.set(cv2.CAP_PROP_POS_MSEC, ctrl["seek"] * 1000.0)
            ctrl["seek"] = None
            tracker.reset()
            sg_filter.reset()
            playback_start = None
            global_state.reset_analytics(mode)
            sa = global_state.session_analytics[mode]

        # 2. Timer
        if mode == "video":
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            stats["time"] = ms / 1000.0 if ms else 0
        else:
            now = time.time()
            if ctrl["state"] == "play":
                if sa.get("last_tick") is not None:
                    sa["elapsed"] += now - sa["last_tick"]
                sa["last_tick"] = now
            else:
                sa["last_tick"] = None
            stats["time"] = sa.get("elapsed", 0)

        # 3. Pause — keep yielding the last frame to keep the connection alive
        if ctrl["state"] == "pause":
            playback_start = None
            if last_jpeg:
                yield Annotator.format_multipart_jpeg(last_jpeg)
            time.sleep(0.1)
            continue

        # 4. Read frame
        ok, frame = cap.read()
        if not ok:
            if isinstance(video_source, str):
                ctrl["state"] = "pause"
                stats["state"] = "Video Ended"
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        # 5. Video real-time skip-ahead
        if mode == "video":
            now_wall = time.time()
            cur_ms   = cap.get(cv2.CAP_PROP_POS_MSEC)
            if playback_start is None:
                playback_start = now_wall
                video_start_ms = cur_ms

            target_ms = video_start_ms + (now_wall - playback_start) * 1000.0
            skipped = 0
            while cur_ms < target_ms and skipped < 30:
                ok, frame = cap.read()
                if not ok:
                    break
                cur_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                skipped += 1
            if not ok:
                ctrl["state"] = "pause"
                stats["state"] = "Video Ended"
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            stats["time"] = cur_ms / 1000.0 if cur_ms else 0

        # Mirror webcam
        if mode == "live":
            frame = cv2.flip(frame, 1)

        sa["total_frames"] += 1

        # 6. Detection
        detections = detector.detect(frame)
        Annotator.draw_detections(frame, detections)
        target = detector.get_target(detections)

        # 7. Kalman predict
        predicted_pos = tracker.predict()

        # 8. Track + Filter
        if target:
            sa["detected_frames"] += 1
            tracker.update(target.center)

            ppm = config.PIXELS_PER_METER
            stats["ball_x"]     = f"{target.center[0] / ppm:.4f}"
            stats["ball_y"]     = f"{target.center[1] / ppm:.4f}"
            stats["confidence"] = f"{target.confidence:.2f}"

            filt = sg_filter.process(target.center[1], fps)
            if filt:
                stats["y_smooth"]  = f"{filt.y_smooth:.4f}"
                stats["v_y"]       = f"{filt.velocity:.4f}"
                stats["accel_y"]   = f"{filt.acceleration:.4f}"
                stats["direction"] = filt.direction
                stats["state"]     = "Tracking"

                if len(sa["all_v_y"]) < config.HISTORY_CAP:
                    sa["all_v_y"].append(filt.velocity)
                    sa["all_accel_y"].append(filt.acceleration)
                    sa["all_y_smooth"].append(filt.y_smooth)

                sa["max_v_y"] = max(sa["max_v_y"], abs(filt.velocity))
                sa["min_v_y"] = min(sa["min_v_y"], filt.velocity)
                stats["max_v_y"] = f"{sa['max_v_y']:.4f}"

                avg = float(np.mean(sa["all_v_y"])) if sa["all_v_y"] else 0
                stats["avg_v_y"] = f"{avg:.4f}"
            else:
                stats["state"] = "Buffering SG..."
        else:
            sa["lost_frames"] += 1
            tracker.mark_lost(predicted_pos)
            stats["state"] = "Lost/Predicting"
            for k in ("ball_x", "ball_y", "confidence", "direction"):
                stats[k] = "--"

        # 9. Annotate trail + prediction
        Annotator.draw_tracker(frame, tracker, predicted_pos)

        # 10. Encode & yield
        jpeg_bytes = Annotator.encode_jpeg(frame)
        if jpeg_bytes:
            last_jpeg = jpeg_bytes          # cache for pause-hold
            yield Annotator.format_multipart_jpeg(jpeg_bytes)


def stream_stats_generator(mode):
    """SSE generator — pushes analytics at ~10 Hz."""
    while True:
        payload = global_state.latest_stats.get(mode, {}).copy()
        payload["play_state"] = global_state.stream_controls[mode]["state"]
        yield f"data: {json.dumps(payload)}\n\n"
        time.sleep(0.1)
