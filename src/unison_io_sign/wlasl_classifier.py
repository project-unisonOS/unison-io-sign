"""
WLASL classifier/translator wrapper using ONNX Runtime.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .keypoints import KeypointResult

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - optional dependency in some environments
    ort = None


class WLASLClassifier:
    def __init__(self, model_path: str, session: Optional[Any] = None, labels_path: Optional[str] = None):
        self.model_path = model_path
        self.session = session or self._load_session(model_path)
        self.labels = self._load_labels(labels_path)

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def _load_session(self, path: str):
        if ort is None:
            return None
        if not os.path.exists(path):
            return None
        try:
            return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        except Exception:
            return None

    def _load_labels(self, labels_path: Optional[str]) -> Dict[int, Dict[str, Any]]:
        if not labels_path or not os.path.exists(labels_path):
            return {}
        try:
            import json

            with open(labels_path, "r") as f:
                data = json.load(f)
            labels_list = data.get("labels", [])
            return {int(item["id"]): {"text": item.get("text", ""), "gloss": item.get("gloss", [])} for item in labels_list}
        except Exception:
            return {}

    def _keypoints_to_features(self, keypoints: KeypointResult) -> np.ndarray:
        """
        Flatten per-frame (x, y, z) coordinates into a single 2D feature tensor [1, N].
        If frame_features are present, use them; otherwise flatten landmarks directly.
        """

        def _flatten_landmarks(landmarks: List[Any]) -> List[float]:
            flat: List[float] = []
            for lm in landmarks:
                if hasattr(lm, "x") and hasattr(lm, "y") and hasattr(lm, "z"):
                    flat.extend([float(lm.x), float(lm.y), float(lm.z)])
                else:
                    try:
                        seq = list(lm)
                        if len(seq) >= 3:
                            flat.extend([float(seq[0]), float(seq[1]), float(seq[2])])
                    except Exception:
                        continue
            return flat

        if keypoints.frame_features:
            flat = [coord for frame in keypoints.frame_features for coord in frame]
        else:
            flat = _flatten_landmarks(keypoints.hand_landmarks) + _flatten_landmarks(keypoints.body_landmarks)
        if not flat:
            flat = [0.0]
        return np.array([flat], dtype=np.float32)

    def predict(self, keypoints: KeypointResult, hint_text: Optional[str] = None) -> Tuple[str, float, List[str]]:
        """
        Return (text, confidence, gloss_list).
        """
        if not self.loaded:
            text = hint_text or "asl_wlasl_stub"
            gloss = [] if hint_text else ["STUB"]
            confidence = 0.9 if hint_text else 0.65
            return text, confidence, gloss

        features = self._keypoints_to_features(keypoints)
        inputs = {self.session.get_inputs()[0].name: features}  # type: ignore[index]
        try:
            outputs = self.session.run(None, inputs)  # type: ignore[call-arg]
            scores = outputs[0].squeeze()
            text = hint_text or "asl_wlasl_onnx"
            gloss: List[str] = [] if hint_text else ["ONNX"]
            confidence = 0.7
            try:
                # If labels exist, compute argmax and map to text/gloss.
                if isinstance(scores, np.ndarray) and scores.ndim >= 1:
                    logits = scores
                    # softmax
                    exp_logits = np.exp(logits - np.max(logits))
                    probs = exp_logits / np.sum(exp_logits)
                    idx = int(np.argmax(probs))
                    confidence = float(np.max(probs))
                    if self.labels and idx in self.labels:
                        text = self.labels[idx].get("text", text)
                        gloss = self.labels[idx].get("gloss", gloss)
                else:
                    confidence = float(scores)
            except Exception:
                pass
            return text, confidence, gloss
        except Exception:
            text = hint_text or "asl_wlasl_stub"
            gloss = [] if hint_text else ["STUB"]
            confidence = 0.7
            return text, confidence, gloss
