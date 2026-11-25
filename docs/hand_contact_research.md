# Hand-contact validation: research, implementation, and measurements

Decision recorded on 10 November 2026. The goal is to reject a ball-motion
reversal caused by a hand holding/moving the golf ball while preserving genuine
hammer taps and limiting additional live-processing work.

## Research comparison

| Work | Relevant finding | Decision for this project |
| --- | --- | --- |
| [MediaPipe Hands, Zhang et al., 2020](https://arxiv.org/html/2006.10214v1) | A palm detector feeds a crop-based model predicting 21 hand landmarks. Its video pipeline avoids redundant full-image detection by reusing tracked crops. | Choose the compact palm/landmark architecture. Finger landmarks give a more useful image-space contact proxy than a broad hand box. Our event-triggered scheduling is an application adaptation, not the complete MediaPipe tracking graph. |
| [On-device Real-time Hand Gesture Recognition, Sung et al., 2021](https://arxiv.org/html/2111.00038v1) | Builds on hand tracking with improved crop alignment, metric hand landmarks, and a separate gesture classifier. | We need landmark geometry rather than gesture labels, so no gesture-classification network is added. Hand-relative depth cannot directly establish the ball's depth. |
| [RTMPose, Jiang et al., 2023](https://arxiv.org/html/2303.07399v2) | Optimizes real-time pose estimation and uses skip-frame detection. Its body and whole-body benchmark results are tied to particular datasets, hardware, and pipeline settings. | A credible alternative, but published body/whole-body FPS is not a hand-contact benchmark. The ready-made compact hand models fit this existing OpenCV project with less integration. No claim is made that MediaPipe outperforms RTMPose in a controlled comparison. |
| [Learning Explicit Contact for Implicit Reconstruction of Hand-held Objects, Hu et al., 2023](https://arxiv.org/abs/2305.20089) | Models 3D hand-object contact explicitly for object reconstruction using learned contact prediction. | True 3D contact is a separate task. Adding that reconstruction pipeline is outside this low-overhead gate; the implementation must describe its contact output as likely image overlap. |

The [OpenCV Zoo hand-model documentation](https://github.com/opencv/opencv_zoo/blob/main/models/handpose_estimation_mediapipe/README.md)
provides converted ONNX weights and cautions that its int8 model can lose substantial
accuracy. This project bundles FP32 weights instead. They are not newly trained
or evaluated for hammer drills by the cited authors.

## Implemented decision

1. Continue ball detection and timestamp-aware motion analysis normally.
2. Save a short bounded history of unannotated crops centered on the ball. No
   hand network runs merely because a video frame arrived.
3. When a reversal passes the motion, acceleration, and cooldown gates, run
   palm detection (192 x 192 input) and hand landmarks (224 x 224 per nearby
   hand). Refine at most two nearby palms. A hand elsewhere in the image does
   not automatically invalidate a tap.
4. Test the ball against the palm polygon and individual finger capsules, with
   a small configurable margin and palm-scaled finger thickness. A rectangular
   hand box or convex hull of all fingertips would also include empty gaps.
5. Inspect the current crop and at most one earlier crop within 120 ms. This
   covers a recent release or smoothing delay without waiting for future frames.
6. Reject likely-contact candidates **before** hit/streak/cooldown registration.
   Store `REJECTED` in a separate audit history, not in the accepted hit timeline.

Models are shared safely behind a lock. ONNX Runtime uses one-thread CPU sessions
when the package is already installed; OpenCV DNN remains the no-extra-package
fallback. No package installation or model download occurs during application
startup or playback. Source revision, weights' checksums, license, and local
adapter changes are retained in `third_party/opencv_zoo/` and `models/hands/`.

## Uncertainty policy

The default rejects established image overlap, not every weak hand prediction.
An unresolved nearby pose is displayed as unclear and accepted motion events are
tagged `hand_uncertain`. Setting `HAND_REJECT_UNCERTAIN=True` excludes those
ambiguous events too, at the cost of potentially discarding genuine strikes.
Missing/broken models do not quietly bypass validation: the candidate is marked
unverified and excluded. No detected hand is not proof that a hand was absent.

Relevant settings are in `config.py`: hand/palm confidence, contact margin,
finger radius, crop size, lookback interval, maximum nearby hands, and strictness.
Seek, pause, replay, and detection loss clear contact continuity. Rejections do
not add hits, increment streaks, or start an accepted-hit cooldown; the existing
timeout may still close a streak.

## Measured performance on this workspace

Command:

```powershell
py -3 scripts/benchmark_hand_contact.py data/videos/Javed.mp4 --output outputs/verification/hands/benchmark_Javed_optimized.json
```

The complete clip has 358 frames. Models were loaded/warmed first; the two passes
exclude annotation drawing, JPEG encoding, browser transport, and playback waits.
The backend was ONNX Runtime 1.24.4, CPU, one thread per hand model.

| Measurement | Observed value |
| --- | ---: |
| Motion-only accepted hits | 22 |
| Guarded accepted hits | 20 |
| Candidates rejected for likely hand contact | 3 |
| Guarded candidate checks | 23 |
| Mean hand validation time per candidate | 56.525 ms |
| Candidate validation p95 | 67.439 ms |
| Hand validation time averaged over all video frames | 3.631 ms/frame |
| Motion-only processing time | 51.872 ms/frame (19.28 FPS) |
| Guarded processing time | 69.410 ms/frame (14.41 FPS) |

The passes ran sequentially under the machine's current load, so the difference
in total pass time is not an isolated causal measurement of the hand network.
The direct hand timers quantify that added work. A prior OpenCV-DNN run averaged
203.705 ms per candidate under its measured conditions; the optimized backend
substantially reduced those spikes. Existing YOLO inference is already a major
cost: this feature does not promise a 30 FPS complete pipeline on this machine.

Candidate totals need not equal the old hit count: a rejected candidate does not
start the accepted-hit cooldown, so another candidate can be examined sooner.
Eleven accepted taps had unclear hand poses, which are explicitly tagged for
review. These counts are predictions, not accuracy scores or proof that every
accepted event was a hammer strike.

## Verification and practical limits

Regression tests cover fingertip, finger-segment, and palm overlap; empty space
inside a hand box; a distant hammer-gripping hand; recent release; stale history;
model failure; uncertainty modes; and rejection before streak accounting. The
frontend displays ignored taps separately and reports check latency. Offline
analysis, HTTP streaming, and sweep replay use the same contact decision logic.

One RGB view cannot distinguish every projected overlap from physical contact.
Motion blur, severe occlusion, a hand outside the crop, missed palm detection,
and ball localization error can still defeat the gate. The repository has no
frame-level hand-contact/strike ground truth, so precision/recall claims would
be unsupported. A reliable accuracy study needs labeled examples of real hammer
taps, hand lifts, catches/releases, near passes without contact, and occlusions.

The implementation is an added false-positive filter, not a verified hammer-head
contact detector or an anti-cheating guarantee.
