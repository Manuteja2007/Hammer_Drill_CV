"""Paper-only timing: detection, Kalman, 2-D SG, velocity and acceleration.

No FrameProcessor, hand model, hit detector, or miss logic is invoked.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from importlib.metadata import version

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
import config
from pipeline.detector import YOLODetector
from pipeline.tracker import KalmanTracker
from pipeline.filter import SavgolFilter
from pipeline.annotator import Annotator


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "p99_ms": float(np.percentile(values, 99))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, default=config.BASE_DIR / "fps_test.mp4")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, help="Optional CPU inference thread count, applied after predictor setup")
    parser.add_argument("--output", type=Path, default=config.BASE_DIR / "outputs/paper_benchmark")
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 1:
        parser.error("runs and warmup must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError("Cannot open benchmark video")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    metadata = {"width": int(cap.get(3)), "height": int(cap.get(4)),
                "source_fps": fps, "metadata_frames": int(cap.get(7))}
    ok, first = cap.read(); cap.release()
    if not ok or fps <= 0:
        raise ValueError("No valid video frames/FPS")
    t = perf_counter()
    detector = YOLODetector(config.MODEL_PATH, config.CONFIDENCE_THRESHOLD, config.TARGET_CLASS)
    load_ms = (perf_counter() - t) * 1000
    # Explicit CPU/FP32, fixed input size, batch one; synchronous CPU timing.
    detector.model.to("cpu")
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("threads must be positive")
        detector.detect(first)
        torch.set_num_threads(args.threads)
    t = perf_counter()
    for _ in range(args.warmup):
        detector.detect(first)
    warmup_ms = (perf_counter() - t) * 1000
    rows, runs = [], []
    motion_path = args.output / "motion_samples.csv"
    with motion_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "video_time_s", "source", "x_m", "y_m", "vx_m_s", "vy_m_s", "ax_m_s2", "ay_m_s2"])
        for repeat in range(args.runs):
            tracker = KalmanTracker()
            filters = [SavgolFilter(config.SG_BUFFER_SIZE, config.SG_POLY_ORDER, config.PIXELS_PER_METER) for _ in range(2)]
            cap = cv2.VideoCapture(str(args.video))
            index = 0
            started = perf_counter()
            try:
                while True:
                    t0 = perf_counter(); ok, frame = cap.read(); t1 = perf_counter()
                    if not ok: break
                    index += 1
                    detections = detector.detect(frame)
                    target = detector.get_target(detections); t2 = perf_counter()
                    prediction = tracker.predict()
                    if target is not None:
                        tracker.update(target.center)
                        point, source = target.center, "measured"
                    else:
                        tracker.mark_lost(prediction)
                        point, source = prediction, "predicted" if prediction is not None else "unavailable"
                    t3 = perf_counter()
                    motion = [filters[axis].process(point[axis], fps, index/fps) for axis in range(2)] if point is not None else [None, None]
                    t4 = perf_counter()
                    Annotator.draw_detections(frame, detections)
                    Annotator.draw_tracker(frame, tracker, prediction)
                    if motion[1] is not None:
                        cv2.putText(frame, f"Vy {motion[1].velocity:.3f} m/s Ay {motion[1].acceleration:.3f} m/s2",
                                    (12, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255), 1)
                    t5 = perf_counter()
                    jpeg = Annotator.encode_jpeg(frame)
                    if jpeg is None: raise RuntimeError("JPEG encoding failed")
                    t6 = perf_counter()
                    row = {"run": repeat+1, "frame": index, "source": source,
                           "decode_ms": (t1-t0)*1000, "detection_ms": (t2-t1)*1000,
                           "tracking_ms": (t3-t2)*1000, "motion_ms": (t4-t3)*1000,
                           "drawing_ms": (t5-t4)*1000, "jpeg_ms": (t6-t5)*1000,
                           "core_ms": (t4-t1)*1000, "local_frame_ms": (t6-t0)*1000}
                    rows.append(row)
                    if repeat == 0:
                        writer.writerow([index, index/fps, source, *([motion[0].y_smooth, motion[1].y_smooth,
                            motion[0].velocity, motion[1].velocity, motion[0].acceleration, motion[1].acceleration]
                            if all(item is not None for item in motion) else [""]*6)])
                elapsed = perf_counter() - started
                if not index: raise ValueError("Empty benchmark pass")
                current = [r for r in rows if r["run"] == repeat+1]
                runs.append({"run": repeat+1, "frames": index, "wall_seconds": elapsed,
                             "wall_fps": index/elapsed,
                             "core_fps": 1000/np.mean([r["core_ms"] for r in current])})
                print(f"Run {repeat+1}: {index} frames, {index/elapsed:.2f} local FPS", flush=True)
            finally:
                cap.release()
    stages = {key: describe([r[key] for r in rows]) for key in rows[0] if key.endswith("_ms")}
    deadline = 1000/fps
    try:
        cpu = subprocess.check_output(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"], text=True).strip()
    except Exception:
        cpu = platform.processor()
    result = {"scope": "Paper-specific 2-D tracked coordinates including predicted gaps; no hand/hit/miss processing",
              "video": {**metadata, "path": str(args.video.resolve()), "sha256": hashlib.sha256(args.video.read_bytes()).hexdigest()},
              "environment": {"cpu": cpu, "logical_cpus": os.cpu_count(), "platform": platform.platform(),
                  "python": platform.python_version(), "packages": {p: version(p) for p in ["torch", "ultralytics", "numpy", "opencv-python"]},
                  "device": "cpu", "precision": "FP32", "batch": 1, "inference_imgsz": 640,
                  "torch_threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads()},
              "model": {"path": str(config.MODEL_PATH), "sha256": hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest(),
                        "confidence": config.CONFIDENCE_THRESHOLD,
                        "training_freeze": detector.model.ckpt.get("train_args", {}).get("freeze"),
                        "effective_imgsz": detector.model.predictor.args.imgsz},
              "settings": {"sg_window": config.SG_BUFFER_SIZE, "sg_degree": config.SG_POLY_ORDER, "pixels_per_metre": config.PIXELS_PER_METER},
              "model_load_ms": load_ms, "warmup_iterations": args.warmup, "warmup_total_ms": warmup_ms,
              "runs": runs, "stages": stages, "source_deadline_ms": deadline,
              "core_fps": 1000/stages["core_ms"]["mean_ms"],
              "local_fps": sum(r["frames"] for r in runs)/sum(r["wall_seconds"] for r in runs),
              "local_frames_within_source_deadline_percent": 100*sum(r["local_frame_ms"]<=deadline for r in rows)/len(rows),
              "first_run_detected_frames": sum(r["source"]=="measured" for r in rows if r["run"]==1),
              "limitations": "Local service-time benchmark, not camera-to-screen latency. No browser/network, capture queue, video writing or pacing. Numeric motion CSV writing on pass one adds to wall time. No ground-truth motion accuracy measured. Paper adapter applies SG to X/Y and Kalman-filled gaps; current dashboard uses measured Y only."}
    with (args.output/"frame_timings.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"core_fps":result["core_fps"],"local_fps":result["local_fps"],"local_mean_ms":stages["local_frame_ms"]["mean_ms"]}),flush=True)


if __name__ == "__main__":
    main()
