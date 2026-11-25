"""Extract measured motion once, then compare configurable hit-detection gates."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from pipeline.processor import FrameProcessor, make_hit_detector
from pipeline.hand_contact import ContactDecision
from pipeline.video import VideoReader
from stream import get_detector

PARAM_SETS = [
    {"label": "No fall gate", "min_fall_velocity": 0.0, "color": "#ef4444"},
    {"label": "Gentle (0.03)", "min_fall_velocity": 0.03, "color": "#f59e0b"},
    {"label": "Configured", "min_fall_velocity": config.HIT_MIN_FALL_VELOCITY, "color": "#22c55e"},
    {"label": "Strict (0.10)", "min_fall_velocity": 0.10, "color": "#3b82f6"},
    {"label": "Very strict (0.15)", "min_fall_velocity": 0.15, "color": "#8b5cf6"},
]


def extract_trajectory(video_path, detector=None):
    processor = FrameProcessor(detector if detector is not None else get_detector())
    data = []
    with VideoReader(video_path) as video:
        for _, timestamp, frame in video:
            result = processor.process(frame, video.fps, timestamp, annotate=False)
            filtered = result.filtered
            # Offline comparison checks every valid sample once so even presets
            # creating additional candidates have matching hand-contact evidence.
            decision = processor.hand_guard.validate() if filtered is not None else None
            data.append({"timestamp": timestamp, "y_smooth": filtered.y_smooth if filtered else None,
                         "velocity": filtered.velocity if filtered else None,
                         "acceleration": filtered.acceleration if filtered else None,
                         "contact_allowed": decision.allowed if decision else None,
                         "contact_reason": decision.reason if decision else None,
                         "contact_sample_timestamp": decision.sample_timestamp if decision else None})
        fps = video.fps
    if not data:
        raise ValueError("The video contains no readable frames")
    return data, fps


def run_detector_on_data(data, fps, parameters):
    options = {key: value for key, value in parameters.items() if key not in {"label", "color"}}
    detector = make_hit_detector(**options)
    for row in data:
        if row["velocity"] is None:
            detector.process_lost_frame(fps, row["timestamp"])
        else:
            def validate():
                return ContactDecision(row["contact_allowed"], row["contact_reason"] == "hand_contact",
                                       row["contact_reason"], "Cached hand check", row["timestamp"],
                                       sample_timestamp=row["contact_sample_timestamp"])
            detector.process(row["y_smooth"], row["velocity"], fps, row["timestamp"], row["acceleration"], validate)
    detector.finalize_session()
    return detector.get_summary()


def create_figure(data, results, parameter_sets, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [row["timestamp"] for row in data]
    fig, axes = plt.subplots(len(results), 3, figsize=(19, 3.5 * len(results)), squeeze=False)
    for index, (parameters, summary) in enumerate(zip(parameter_sets, results)):
        hits = [event for event in summary["timeline"] if event["type"] == "HIT"]
        for column, (key, title) in enumerate([("y_smooth", "Vertical position (m)"),
                                                ("velocity", "Velocity (m/s)"),
                                                ("acceleration", "Acceleration (m/s2)")]):
            axis = axes[index, column]
            # None values preserve gaps instead of drawing a line through lost tracking.
            axis.plot(times, [row[key] for row in data], color=parameters["color"], linewidth=0.8)
            for event in summary["timeline"]:
                axis.axvline(event["timestamp"], color="green" if event["type"] == "HIT" else "red", alpha=0.4)
            axis.set_title(title)
            axis.set_xlabel("Time (s)")
            axis.grid(alpha=0.2)
            if column == 0:
                axis.invert_yaxis()
                axis.set_ylabel(f"{parameters['label']}\n{len(hits)} hits, {summary['rejected_count']} rejected")
            elif column == 1:
                axis.axhline(config.HIT_VELOCITY_THRESHOLD, color="gray", linestyle=":")
                axis.axhline(-config.HIT_VELOCITY_THRESHOLD, color="gray", linestyle=":")
    fig.suptitle("Hit detection sweep (hand-contact checks and configured acceleration gating)")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, default=config.DEFAULT_VIDEO_PATH)
    parser.add_argument("--output", type=Path, default=config.OUTPUT_FOLDER / "plots" / "hit_detection_sweep.png")
    args = parser.parse_args()
    try:
        data, fps = extract_trajectory(args.video)
        results = [run_detector_on_data(data, fps, parameters) for parameters in PARAM_SETS]
        output = create_figure(data, results, PARAM_SETS, args.output)
        output.with_suffix(".json").write_text(json.dumps([
            {"parameters": parameters, "summary": summary}
            for parameters, summary in zip(PARAM_SETS, results)
        ], indent=2), encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    for parameters, summary in zip(PARAM_SETS, results):
        print(f"{parameters['label']}: {summary['total_hits']} hits, {summary['total_misses']} misses")
    print(f"Comparison: {output}")


if __name__ == "__main__":
    main()
