# ASL Model Setup (Local-first)

This is the plan for wiring the ASL provider to a local model using keypoints → WLASL-based classifier/translator.

## Components
- Keypoint extraction: MediaPipe Hands/BlazePose (or MMPose). Industry-standard and hardware-optimized.
- Recognition/translation: WLASL-based transformer/classifier (OpenMMLab/MMACTION or similar).

## Expected runtime bits (language-aware)
- `UNISON_SIGN_LANGUAGE` → default `asl`.
- `UNISON_SIGN_MODEL_PATH_<LANG>` → per-language checkpoint, e.g. `UNISON_SIGN_MODEL_PATH_ASL`.
- `UNISON_SIGN_MODEL_PATH` → generic fallback if per-language is not set.
- Optional: `UNISON_SIGN_KEYPOINT_BACKEND_<LANG>` or `UNISON_SIGN_KEYPOINT_BACKEND` → `mediapipe` (default target) or `mmpose`.
- GPU/accelerator recommended; CPU fallback allowed but slower.

## Installing runtime dependencies (dev)
```bash
pip install mediapipe onnxruntime numpy  # or onnxruntime-gpu if available
# optional alternative:
# pip install mmpose mmengine torch torchvision
```

## Wiring plan
- Update `ASLProvider` to:
  - Load a keypoint extractor (MediaPipe or MMPose).
  - Load the WLASL classifier/translator from `UNISON_SIGN_MODEL_PATH_<LANG>` (ONNX via onnxruntime).
  - Run keypoints → gloss/text/intent hints.
- Add replay fixtures:
  - Precomputed keypoints for a few commands (e.g., OPEN_SETTINGS, OPEN_APP).
  - Validate outputs via integration tests.

## Privacy / locality
- All video/keypoints remain on-device unless explicitly configured otherwise.
- Model downloads should be cached locally; no runtime fetches without consent.
