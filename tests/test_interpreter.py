from dataclasses import dataclass

from unison_io_sign.interpreter import SignInterpreter, InterpreterConfig
from unison_io_sign.providers.asl import ASLProvider


@dataclass
class FakeFrame:
    sign_likelihood: float = 0.0
    timestamp_ms: int = 0


def test_interpreter_batches_frames():
    provider = ASLProvider()
    interpreter = SignInterpreter(provider, InterpreterConfig(segment_size=3))
    frames = [FakeFrame() for _ in range(5)]
    interpretations = interpreter.ingest_frames(frames)
    # 3 frames trigger first segment, remaining 2 are buffered
    assert len(interpretations) == 1
    # Flush should produce final segment
    flushed = interpreter.flush()
    assert len(flushed) == 1
    # Provider stub returns language code and zero confidence
    assert interpretations[0].language == "asl"
    assert flushed[0].language == "asl"
