# Implementation notes

## Data flow

A signed Flask cookie identifies an in-memory `BrowserSession`. Its live and video modes each own a `StreamController`, an `AnalyticsState`, and a `FrameProcessor`. The YOLO model loads lazily and is shared behind a predictor lock. The camera has a separate exclusive ownership lock.

The processor detects before drawing, selects the ball, updates the Kalman trail, fits a local polynomial to measured vertical positions, and passes velocity plus acceleration into the hit detector. It produces the same metrics in the web app and offline analyzer. Predicted positions are only for drawing; no synthetic measurement is silently included in derivatives.

Before a motion-valid reversal is registered, the hit detector invokes the processor's hand-contact validator. It inspects a bounded history of unannotated ball-centered crops using MediaPipe palm/hand models. Current and recent image-space contact rejects the candidate before any hit, streak, or accepted-hit cooldown is updated. Rejections have a separate audit history. A weak pose remains distinct from contact; optional strict handling is configurable. ONNX Runtime uses a one-thread CPU session if installed; otherwise the same transforms use OpenCV DNN. Model adapters/weights are bundled, attributed, and never downloaded at runtime.

MJPEG carries annotated images. A separate SSE connection publishes numeric metrics at 10 Hz. A stream's lock protects processing and all controls; it is released before network yields or playback waits. Stop finalizes immediately under that lock. Each generator owns a token so an old response cannot stop a newer stream. The response close callback also handles a stream that never starts iterating.

## HTTP interface

| Method | Route | Behavior |
| --- | --- | --- |
| GET | `/` | Dashboard |
| POST | `/upload_video` | Multipart `video`; validates extension, size, and decodability |
| GET | `/check_video_uploaded` | Whether this browser has a valid upload |
| GET | `/video_feed_live` | Open webcam MJPEG stream |
| GET | `/video_feed_file` | Open this browser's uploaded video stream |
| POST | `/video_control` | JSON `mode`, `action`; actions: play, pause, seek, stop |
| GET | `/stream_stats/<mode>` | SSE measurements and playback state |
| GET | `/session_summary/<mode>` | Snapshot of accumulated and finalized results |

Modes are `live` and `video`. Seek accepts finite seconds within the source duration and only applies to video. Invalid requests receive JSON errors. Missing uploads return 404; conflicting streams/camera ownership return 409; unreadable input returns 422; model load failure returns 503. Initial stream failures are surfaced in the UI with a retry message.

## Lifecycle rules

Opening a stream initializes fresh metrics. Pause preserves elapsed activity time and clears derivative/reversal continuity. Seek resets every metric and the tracking/detection state. Replay at EOF opens a fresh capture and starts at zero. EOF preserves the final timestamp and releases the capture. Stop, disconnect, and EOF close active streaks once, without creating a MISS event. Timeout misses require an existing streak.

The session registry expires idle sessions after 24 hours and caps retained sessions at 128, evicting only inactive sessions. Generated upload files stay in `uploads/` for local inspection and can be removed when the app is stopped. Original input media is never stored there or automatically deleted.

## Scope

This is a local single-process application, not a deployed multi-user service. Browser cookie sessions are separate; tabs in one browser share state. Deployment across processes requires persistent/shared state and an explicit camera/inference service. The repository does not contain model training data annotations or verified hit/miss labels.
