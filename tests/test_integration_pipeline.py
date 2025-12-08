from dataclasses import dataclass

from unison_io_sign.detector import SignPresenceDetector, DetectionConfig
from unison_io_sign.interpreter import SignInterpreter, InterpreterConfig
from unison_io_sign.providers.asl import ASLProvider


@dataclass
class FakeFrame:
    sign_likelihood: float
    timestamp_ms: int = 0


def test_pipeline_presence_then_interpretation():
    detector = SignPresenceDetector(DetectionConfig(detect_threshold=0.5, lose_threshold=0.4, sustain_frames=2))
    interpreter = SignInterpreter(ASLProvider(), InterpreterConfig(segment_size=2))

    frames = [FakeFrame(0.6), FakeFrame(0.7), FakeFrame(0.2)]
    events = detector.process_frames(frames)
    assert any(e.event_type == "sign_presence_detected" for e in events)

    interpretations = interpreter.ingest_frames(frames)
    assert len(interpretations) == 1
    assert interpretations[0].language == "asl"
