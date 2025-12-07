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
        # Placeholder interpretation; future: call model to fill text/intent/gloss.
        return SignInterpretation.from_stub(
            language=self.language_code,
            text="",
            intent=None,
            confidence=0.0,
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
