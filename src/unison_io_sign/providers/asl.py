from __future__ import annotations

import os
from typing import List, Optional

from ..provider import SignLanguageProvider
from ..schemas import SignInterpretation, SigningOutput, VideoSegment, AvatarInstructions
from ..keypoints import make_extractor, KeypointResult
from ..wlasl_classifier import WLASLClassifier


class ASLProvider(SignLanguageProvider):
    """
    ASL provider skeleton.

    Phase 2 behavior:
    - If a hint is present in segment.metadata, echo it as text with higher confidence.
    - If a model path is configured, attempt to run the model (stub hook for now).
    - Otherwise return low-confidence empty text.

    Later revisions will load a real local model (keypoints → gloss/text/intent).
    """

    def __init__(self, extractor=None, classifier=None):
        self.model_path = os.getenv("UNISON_ASL_MODEL_PATH")
        self.backend = os.getenv("UNISON_ASL_KEYPOINT_BACKEND", "mediapipe")
        self.extractor = extractor
        self.classifier = classifier
        if self.model_path and self.classifier is None:
            self.classifier = WLASLClassifier(self.model_path)
        if self.extractor is None:
            self.extractor = make_extractor(self.backend)

    @property
    def language_code(self) -> str:
        return "asl"

    def interpret_segment(self, segment: VideoSegment) -> SignInterpretation:
        hint_text = segment.metadata.get("text_hint") if segment.metadata else None
        if self._can_run_model():
            return self._infer_with_model(segment, hint_text=hint_text)

        text = hint_text or ""
        confidence = 0.75 if hint_text else 0.2
        return SignInterpretation.from_stub(
            language=self.language_code,
            text=text,
            intent=None,
            confidence=confidence,
            gloss=[],
            segment=segment,
        )

    def generate_output(self, text: str, gloss: Optional[List[str]] = None) -> SigningOutput:
        return SigningOutput(
            language=self.language_code,
            text=text,
            gloss=gloss or [],
            avatar_instructions=AvatarInstructions(),
        )

    def _can_run_model(self) -> bool:
        return bool(self.classifier) and getattr(self.classifier, "loaded", False)

    def _infer_with_model(self, segment: VideoSegment, hint_text: Optional[str] = None) -> SignInterpretation:
        """
        Placeholder model inference.
        Future: run keypoint/pose → gloss/text model.
        """
        frames = segment.frames or []
        keypoints: KeypointResult = self.extractor.extract(frames) if self.extractor else KeypointResult([], [])
        text, confidence, gloss = self.classifier.predict(keypoints, hint_text=hint_text)  # type: ignore
        return SignInterpretation.from_stub(
            language=self.language_code,
            text=text,
            intent=None,
            confidence=confidence,
            gloss=gloss,
            segment=segment,
        )
