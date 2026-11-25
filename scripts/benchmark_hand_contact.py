"""Compare motion-only and guarded processing on a local video; never install packages."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import config
from pipeline.hand_contact import HandContactGuard, get_hand_detector
from pipeline.processor import FrameProcessor
from pipeline.video import VideoReader
from stream import get_detector


def benchmark(video_path):
    detector = get_detector()
    # Separate cold model loading/warmup from steady-state per-frame measurements.
    with VideoReader(video_path) as reader:
        _, timestamp, frame = next(iter(reader))
        target = detector.get_target(detector.detect(frame))
    hand_model = get_hand_detector()
    if target is not None:
        warmup = HandContactGuard()
        warmup.observe(frame, target, timestamp)
        warmup.validate()
    results = {"hand_backend": hand_model.backend}
    for enabled in (False, True):
        processor = FrameProcessor(detector, hand_guard=HandContactGuard(enabled=enabled))
        frame_times, check_times = [], []
        with VideoReader(video_path) as reader:
            for _, timestamp, frame in reader:
                started = perf_counter()
                result = processor.process(frame, reader.fps, timestamp, annotate=False)
                frame_times.append((perf_counter() - started) * 1000)
                if enabled and result.hand_check is not None:
                    check_times.append(result.hand_check.latency_ms)
        processor.finalize()
        results["hand_guard" if enabled else "motion_only"] = {
            "summary": processor.analytics.summary(),
            "processing_ms_per_frame": round(float(np.mean(frame_times)), 3),
            "processing_fps": round(1000 / float(np.mean(frame_times)), 2),
            "frame_p95_ms": round(float(np.percentile(frame_times, 95)), 3),
            "mean_candidate_check_ms": round(float(np.mean(check_times)), 3) if check_times else 0.0,
            "candidate_check_p95_ms": round(float(np.percentile(check_times, 95)), 3) if check_times else 0.0,
            "hand_ms_amortized_per_frame": round(sum(check_times) / len(frame_times), 3),
            "frames": len(frame_times),
        }
    results["notes"] = ("Same source processed twice; no JPEG drawing, video pacing, or browser. "
                         "Timings include the candidate hand gate after model warmup. "
                         "Counts are predictions, not ground-truth accuracy measurements.")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, default=config.DEFAULT_VIDEO_PATH)
    parser.add_argument("--output", type=Path, default=config.OUTPUT_FOLDER / "reports" / "hand_contact_benchmark.json")
    args = parser.parse_args()
    result = benchmark(args.video)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    for mode in ("motion_only", "hand_guard"):
        data = result[mode]
        print(f"{mode}: {data['summary']['total_hits']} hits; {data['summary']['rejected_count']} rejected; "
              f"{data['processing_ms_per_frame']} ms/frame; hand cost {data['hand_ms_amortized_per_frame']} ms/frame")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
