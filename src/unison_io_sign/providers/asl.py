from __future__ import annotations

from typing import List, Optional

from ..provider import SignLanguageProvider
from ..schemas import SignInterpretation, SigningOutput, VideoSegment, AvatarInstructions


class ASLProvider(SignLanguageProvider):
    """
    Stub ASL provider for Phase 0.

    Later revisions will hook into a real local model for pose/keypoint-based interpretation.
    """

    @property
    def language_code(self) -> str:
        return "asl"

    def interpret_segment(self, segment: VideoSegment) -> SignInterpretation:
        """
        Phase 2 stub:
        - If a hint is present in segment.metadata, echo it as text.
        - Otherwise return an empty text with low confidence.
        """
        hint_text = None
        if segment.metadata:
            hint_text = segment.metadata.get("text_hint")
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
