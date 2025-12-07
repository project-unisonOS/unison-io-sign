"""
Shared schemas for sign-language I/O services.

These dataclasses are JSON-friendly and intentionally minimal for Phase 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import uuid
import time

JsonDict = Dict[str, Any]


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class AvatarInstructions:
    version: str = "1.0"
    rig: str = "default_humanoid"
    keyframes: List[JsonDict] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class SigningOutput:
    language: str
    text: str
    gloss: Optional[List[str]] = None
    avatar_instructions: AvatarInstructions = field(default_factory=AvatarInstructions)

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        # Ensure nested dataclass is flattened correctly
        data["avatar_instructions"] = self.avatar_instructions.to_dict()
        return data


@dataclass
class SignPresenceEvent:
    event_type: str  # "sign_presence_detected" | "sign_presence_lost"
    timestamp: str
    source: str
    session_id: Optional[str] = None
    language_hint: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class VideoSegment:
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time_ms: int = field(default_factory=_now_ms)
    end_time_ms: Optional[int] = None
    frames: Optional[List[Any]] = None  # placeholder; future: keypoints or frame refs
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class SignInterpretation:
    language: str
    segment_id: str
    start_time_ms: int
    end_time_ms: int
    confidence: float
    type: str = "utterance"  # utterance | command | gesture
    text: Optional[str] = None
    intent: Optional[JsonDict] = None
    raw_gloss: Optional[List[str]] = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_stub(
        cls,
        language: str,
        text: str,
        intent: Optional[JsonDict] = None,
        confidence: float = 0.5,
        gloss: Optional[List[str]] = None,
        segment: Optional[VideoSegment] = None,
    ) -> "SignInterpretation":
        seg = segment or VideoSegment()
        return cls(
            language=language,
            segment_id=seg.segment_id,
            start_time_ms=seg.start_time_ms,
            end_time_ms=seg.end_time_ms or _now_ms(),
            confidence=confidence,
            text=text,
            intent=intent,
            raw_gloss=gloss,
        )
