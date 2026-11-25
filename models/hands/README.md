# Hand-contact models

These FP32 MediaPipe palm and 21-landmark models are distributed by OpenCV Zoo
under Apache-2.0. The complete license is in `third_party/opencv_zoo/LICENSE`.

`manifest.json` records the pinned source revision URLs, file sizes, and SHA-256
checksums. Application startup does not download weights or install packages.

The hand landmark model's upstream documentation cautions that its int8 version
can lose substantial accuracy. This project uses the FP32 models instead.

Source: https://github.com/opencv/opencv_zoo/tree/main/models/handpose_estimation_mediapipe
