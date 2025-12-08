from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .schemas import SignInterpretation, VideoSegment
from .provider import SignLanguageProvider


@dataclass
class InterpreterConfig:
    segment_size: int = 8  # frames per segment for Phase 1 stub
    language_code: str = "asl"


class SignInterpreter:
    """
    Segmentation + provider wiring skeleton.

    Phase 1: batches frames into fixed-size segments and calls the configured provider.
    """

    def __init__(self, provider: SignLanguageProvider, config: Optional[InterpreterConfig] = None):
        self.provider = provider
        self.config = config or InterpreterConfig()
        self._buffer: List[object] = []

    def ingest_frames(self, frames: Iterable[object]) -> List[SignInterpretation]:
        interpretations: List[SignInterpretation] = []
        for frame in frames:
            self._buffer.append(frame)
            if len(self._buffer) >= self.config.segment_size:
                segment = self._flush_segment()
                interp = self.provider.interpret_segment(segment)
                interpretations.append(interp)
        return interpretations

    def flush(self) -> List[SignInterpretation]:
        """Flush any residual frames into one last segment if present."""
        if not self._buffer:
            return []
        segment = self._flush_segment()
        return [self.provider.interpret_segment(segment)]

    def _flush_segment(self) -> VideoSegment:
        frames = list(self._buffer)
        self._buffer.clear()
        return VideoSegment(frames=frames)
