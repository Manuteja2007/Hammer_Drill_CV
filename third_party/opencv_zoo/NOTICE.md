# OpenCV Zoo hand models and adapters

Source: https://github.com/opencv/opencv_zoo

The exact source revision is recorded in `REVISION`. The two adapters and
MediaPipe-converted ONNX weights are distributed under Apache-2.0 (see `LICENSE`).
The weights' source URLs, sizes, and SHA-256 hashes are recorded in
`models/hands/manifest.json` at the project root.

Local changes to `mp_palmdet.py`: generate the equivalent anchor array instead of
storing thousands of literal coordinates; pass xywh boxes to OpenCV NMS; clip
sigmoid input for numerical stability. Both adapters accept an injected inference
backend so the same geometric transforms can run with ONNX Runtime or OpenCV DNN.

Application-specific contact geometry, temporal validation, and hit rejection
are implemented separately in `pipeline/hand_contact.py`.
