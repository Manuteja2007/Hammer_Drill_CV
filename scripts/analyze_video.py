"""Analyze a video with the same processor as the web dashboard."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import config
from pipeline.processor import FrameProcessor
from pipeline.video import VideoReader
from stream import get_detector


def analyze_video(video_path, output_dir=None, save_video=True, max_frames=None, detector=None):
    video_path = Path(video_path)
    output_dir = Path(output_dir) if output_dir else config.OUTPUT_FOLDER / "runs" / (
        f"{video_path.stem}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    )
    processor = FrameProcessor(detector if detector is not None else get_detector())
    writer = None
    with VideoReader(video_path) as video:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with (output_dir / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
                csv_writer = csv.writer(handle)
                csv_writer.writerow(["frame", "time_s", "detected", "ball_x", "ball_y", "y_smooth",
                                     "velocity", "accel", "direction", "event_type", "streak",
                                     "total_hits", "total_misses", "hand_check", "hand_check_ms",
                                     "rejection_reason", "rejected_total"])
                for index, timestamp, frame in video:
                    result = processor.process(frame, video.fps, timestamp, annotate=save_video)
                    values = processor.analytics.stats
                    csv_writer.writerow([index, timestamp, int(result.target is not None),
                                         values["ball_x"], values["ball_y"], values["y_smooth"],
                                         values["v_y"], values["accel_y"], values["direction"],
                                         result.event.type if result.event else "", values["current_streak"],
                                         values["hit_count"], values["miss_count"],
                                         result.hand_check.status if result.hand_check else "not_checked",
                                         result.hand_check.latency_ms if result.hand_check else "",
                                         result.event.reason if result.event and result.event.type == "REJECTED" else "",
                                         values["rejected_count"]])
                    if save_video:
                        if writer is None:
                            height, width = frame.shape[:2]
                            writer = cv2.VideoWriter(str(output_dir / "annotated.mp4"),
                                                     cv2.VideoWriter_fourcc(*"mp4v"), video.fps, (width, height))
                            if not writer.isOpened():
                                raise RuntimeError("Could not create the annotated output video")
                        writer.write(frame)
                    if max_frames is not None and index >= max_frames:
                        break
            if not processor.analytics.total_frames:
                raise ValueError("The video contains no readable frames")
        finally:
            processor.finalize()
            if writer is not None:
                writer.release()
    summary = processor.analytics.summary()
    summary["source"] = str(video_path.resolve())
    summary["partial"] = max_frames is not None and summary["total_frames"] < video.frame_count
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary, output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, default=config.DEFAULT_VIDEO_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-video", action="store_true", help="Only write measurements and summary")
    parser.add_argument("--max-frames", type=int, help="Limit processing for a quick smoke check")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    try:
        summary, directory = analyze_video(args.video, args.output_dir, not args.no_video, args.max_frames)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(summary, indent=2))
    print(f"Results: {directory}")


if __name__ == "__main__":
    main()
