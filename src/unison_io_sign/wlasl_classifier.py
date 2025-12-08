from __future__ import annotations

"""
Placeholder WLASL classifier/translator wrapper.

In Phase 2 we only verify wiring; real model loading/inference will be added in Phase 3.
"""

import os
from typing import Any, List, Optional, Tuple

from .keypoints import KeypointResult


class WLASLClassifier:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._loaded = self._load_model(model_path)

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _load_model(self, path: str) -> bool:
        # Placeholder: verify path exists. Later: load Torch/ONNX model.
        return os.path.exists(path)

    def predict(self, keypoints: KeypointResult, hint_text: Optional[str] = None) -> Tuple[str, float, List[str]]:
        """
        Return (text, confidence, gloss_list).

        Phase 2: echo hint or fallback to stub text.
        """
        text = hint_text or "asl_wlasl_stub"
        gloss = [] if hint_text else ["STUB"]
        confidence = 0.9 if hint_text else 0.65
        return text, confidence, gloss
