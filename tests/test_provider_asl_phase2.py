from unison_io_sign.providers.asl import ASLProvider
from unison_io_sign.schemas import VideoSegment


def test_asl_provider_uses_text_hint():
    provider = ASLProvider()
    segment = VideoSegment(metadata={"text_hint": "open settings"})
    interp = provider.interpret_segment(segment)
    assert interp.text == "open settings"
    assert interp.confidence == 0.75
