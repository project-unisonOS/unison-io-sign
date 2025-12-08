# Real ASL Model Plug-In Guide

This repo already supports ONNX-based ASL models via `UNISON_SIGN_MODEL_PATH_<LANG>` and `UNISON_SIGN_LABELS_PATH_<LANG>`. To swap in a real WLASL/ASL checkpoint:

## 1) Drop your model and labels
- Place your ONNX model somewhere accessible on disk, e.g. `/models/wlasl_asl.onnx`.
- Create a label map JSON that matches model logits, e.g.:
```json
{
  "labels": [
    { "id": 0, "text": "open settings", "gloss": ["OPEN", "SETTINGS"] },
    { "id": 1, "text": "open browser", "gloss": ["OPEN", "BROWSER"] }
  ]
}
```
- Save it at `/models/wlasl_labels_asl.json` (path is arbitrary; referenced via env var).

## 2) Set environment variables
- `UNISON_SIGN_LANGUAGE=asl` (default)
- `UNISON_SIGN_MODEL_PATH_ASL=/models/wlasl_asl.onnx`
- `UNISON_SIGN_LABELS_PATH_ASL=/models/wlasl_labels_asl.json`
- Optional: `UNISON_SIGN_KEYPOINT_BACKEND_ASL=mediapipe` (default) or `mmpose`

## 3) Run tests with the real model
```bash
cd unison-io-sign
UNISON_SIGN_MODEL_PATH_ASL=/models/wlasl_asl.onnx \
UNISON_SIGN_LABELS_PATH_ASL=/models/wlasl_labels_asl.json \
make test
```

## 4) Replay fixtures
- Add real keypoint dumps under `tests/fixtures/asl/` (e.g., `keypoints_real_clip.json` with `"frames": [[...coords...], ...]`).
- Add a label map alongside or point to the global label JSON.
- Create a new test that loads the real keypoints, runs `ASLProvider`, and asserts the expected text/gloss/confidence.

## Notes
- The classifier uses onnxruntime with CPUExecutionProvider by default; adjust providers if you have GPU/accelerator.
- If the model outputs logits with a different shape, update `WLASLClassifier.predict` accordingly and ensure the label map matches output indices.
