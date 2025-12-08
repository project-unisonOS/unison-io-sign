# ASL Model Setup (Local-first)

This is the plan for wiring the ASL provider to a local model using keypoints → WLASL-based classifier/translator.

## Components
- Keypoint extraction: MediaPipe Hands/BlazePose (or MMPose). Industry-standard and hardware-optimized.
- Recognition/translation: WLASL-based transformer/classifier (OpenMMLab/MMACTION or similar).

## Expected runtime bits
- `UNISON_ASL_MODEL_PATH` → path to a local checkpoint (Torch or ONNX).
- Optional: `UNISON_ASL_KEYPOINT_BACKEND` → `mediapipe` (default target) or `mmpose`.
- GPU/accelerator recommended; CPU fallback allowed but slower.

## Installing runtime dependencies (dev)
```bash
pip install mediapipe torch  # or torch+cuda if available
# optional alternative:
# pip install mmpose mmengine torch torchvision
```

## Wiring plan
- Update `ASLProvider` to:
  - Load a keypoint extractor (MediaPipe or MMPose).
  - Load the WLASL classifier/translator from `UNISON_ASL_MODEL_PATH`.
  - Run keypoints → gloss/text/intent hints.
- Add replay fixtures:
  - Precomputed keypoints for a few commands (e.g., OPEN_SETTINGS, OPEN_APP).
  - Validate outputs via integration tests.

## Privacy / locality
- All video/keypoints remain on-device unless explicitly configured otherwise.
- Model downloads should be cached locally; no runtime fetches without consent.
