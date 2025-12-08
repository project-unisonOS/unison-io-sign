from dataclasses import dataclass

from unison_io_sign.detector import SignPresenceDetector, DetectionConfig


@dataclass
class FakeFrame:
    sign_likelihood: float
    timestamp_ms: int = 0


def test_detector_emits_detect_and_lost():
    detector = SignPresenceDetector(DetectionConfig(detect_threshold=0.6, lose_threshold=0.4, sustain_frames=2))
    frames = [
        FakeFrame(0.2),
        FakeFrame(0.3),
        FakeFrame(0.7),
        FakeFrame(0.8),
        FakeFrame(0.3),
        FakeFrame(0.2),
    ]
    events = detector.process_frames(frames)
    types = [e.event_type for e in events]
    assert "sign_presence_detected" in types
    assert "sign_presence_lost" in types
