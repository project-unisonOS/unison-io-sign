from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol
import time

from .schemas import SignPresenceEvent


class Frame(Protocol):
    """Minimal protocol for frames used in Phase 1 tests."""

    sign_likelihood: float
    timestamp_ms: int


@dataclass
class DetectionConfig:
    detect_threshold: float = 0.6
    lose_threshold: float = 0.3
    sustain_frames: int = 3
    language_hint: str = "asl"
    source: str = "unison-io-sign-detector"


class SignPresenceDetector:
    """
    Lightweight detector skeleton.

    Phase 1: uses a simple likelihood field on incoming frames.
    Later: plug in a real model using keypoints or raw frames.
    """

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()
        self._active = False
        self._buffer: List[Frame] = []

    def _emit_event(self, event_type: str, confidence: float) -> SignPresenceEvent:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return SignPresenceEvent(
            event_type=event_type,
            timestamp=ts,
            source=self.config.source,
            language_hint=self.config.language_hint,
            confidence=confidence,
        )

    def process_frames(self, frames: Iterable[Frame]) -> List[SignPresenceEvent]:
        events: List[SignPresenceEvent] = []
        for frame in frames:
            self._buffer.append(frame)
            # keep small buffer
            if len(self._buffer) > self.config.sustain_frames:
                self._buffer.pop(0)
            avg = sum(f.sign_likelihood for f in self._buffer) / len(self._buffer)
            if not self._active and avg >= self.config.detect_threshold:
                self._active = True
                events.append(self._emit_event("sign_presence_detected", avg))
            elif self._active and avg <= self.config.lose_threshold:
                self._active = False
                events.append(self._emit_event("sign_presence_lost", avg))
        return events
