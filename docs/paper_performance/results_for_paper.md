# Paper performance evidence

Neither tested configuration sustained the source frame rate. The measurements establish recorded-video motion-processing performance on this hardware, but do not substantiate full-frame-rate real-time operation.

## Ready-to-insert subsection (Section IV.D)

Computational performance was evaluated using a 478 x 850 video containing 476 frames at 29.39 FPS on an Intel Core i5-1135G7 CPU (4 cores, 8 logical processors; approximately 8 GB RAM). CPU FP32 inference used batch size one and an image-size setting of 640. After 20 warm-up iterations, three complete passes were timed using a monotonic high-resolution clock. The evaluated pipeline comprised YOLO detection, Kalman tracking, independent second-order seven-sample polynomial filtering of tracked X and Y coordinates, and first- and second-order derivatives for velocity and acceleration. The fastest tested configuration used 4 inference thread(s), yielding 16.68 FPS for detection through motion estimation and 15.77 FPS when decoding, annotation, JPEG encoding and local loop overhead were included. Mean and 95th-percentile local frame service times were 63.41 and 77.42 ms, respectively, compared with a source-frame budget of 34.02 ms. Neither tested configuration sustained the source frame rate. The measurements establish recorded-video motion-processing performance on this hardware, but do not substantiate full-frame-rate real-time operation. Browser rendering, network transfer and camera buffering were excluded.

## Measured configurations

| Configuration | Torch threads | Core FPS | Local FPS | Mean local ms | P95 local ms | Within source deadline (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Default CPU | 4 | 16.68 | 15.77 | 63.41 | 77.42 | 0.0 |
| One-thread CPU | 1 | 12.81 | 12.40 | 80.62 | 100.34 | 0.0 |

Core FPS = 1000 / mean detection-through-motion time. Local FPS = total frames / total measured loop wall time. Do not average instantaneous per-frame FPS.

## Stage timings for fastest tested configuration

| Stage | Mean ms | P95 ms |
| --- | ---: | ---: |
| decode_ms | 1.480 | 1.976 |
| detection_ms | 59.493 | 72.660 |
| tracking_ms | 0.056 | 0.080 |
| motion_ms | 0.388 | 0.517 |
| drawing_ms | 0.206 | 0.344 |
| jpeg_ms | 1.792 | 2.249 |
| core_ms | 59.937 | 73.443 |
| local_frame_ms | 63.415 | 77.423 |

## Repetitions and provenance

Default: run 1: 15.98 FPS (476 frames, 29.80 s), run 2: 15.35 FPS (476 frames, 31.01 s), run 3: 15.99 FPS (476 frames, 29.77 s)
One-thread: run 1: 12.93 FPS (476 frames, 36.81 s), run 2: 11.74 FPS (476 frames, 40.54 s), run 3: 12.59 FPS (476 frames, 37.80 s)

Software: 3.11.8; {'torch': '2.11.0', 'ultralytics': '8.4.31', 'numpy': '2.2.6', 'opencv-python': '4.11.0.86'}. Platform: Windows-10-10.0.26200-SP0. OpenCV threads: 1.
Video SHA-256: 00e88254ff514cc611da93ce39f6f15dfa1018ce38d9b0568675ce77bd4d7827. Model SHA-256: 557c24f525b400714a8664c928d6ced57dd9711a53b492e1ea44180f1835d6f0.
Model checkpoint training freeze field: 10. Verify the exact checkpoint against the paper's six-frozen-module description; a filename is not proof of training provenance.

## Required qualifications before publication

- The supplied clip is 478 x 850, not the paper dataset's stated 720 x 1280. Do not describe these timings as measured at the larger resolution.
- This is a separate paper-specific adapter: SG runs on both tracked axes, including Kalman-filled gaps. The dashboard currently computes measured Y derivatives only. Neither the hand checker nor the hit/miss processor is executed in this benchmark.
- Local service time is not capture-to-display latency. Camera acquisition queues, network transmission, SSE and browser paint require a separate instrumented experiment.
- The pixel-to-metre factor is inherited (2650). Velocity/acceleration values are estimates; physical accuracy and calibration for this clip have not been validated.
- Counts of detected/predicted frames describe availability, not detection accuracy. Numeric samples include a provenance column; predictions are not ground truth.
- The causal fit starts after three samples and grows to seven. It uses past samples only. Startup buffering is separate from computational service time; no full-window future-frame delay is imposed.
- Do not use throughput results to infer the comparative speed of the other YOLO models in Table I; only the supplied weight file was benchmarked.
- Keep the unmodified source PDF. Insert the supplied LaTeX subsection/table into the manuscript source; the PDF was not rewritten or reflowed.

## Reproduce

```powershell
py -3 scripts/benchmark_paper.py
py -3 scripts/benchmark_paper.py --threads 1 --output outputs/paper_benchmark_threads1
py -3 scripts/build_paper_report.py
```
