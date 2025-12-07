from unison_io_sign.provider import register_provider, get_provider
from unison_io_sign.providers.asl import ASLProvider
from unison_io_sign.schemas import VideoSegment


def test_register_and_resolve_provider():
    provider = ASLProvider()
    register_provider(provider)
    resolved = get_provider("asl")
    assert resolved.language_code == "asl"


def test_asl_provider_output_schema():
    provider = ASLProvider()
    segment = VideoSegment()
    interp = provider.interpret_segment(segment)
    assert interp.language == "asl"
    assert interp.segment_id == segment.segment_id
    assert interp.confidence == 0.0
    signing_output = provider.generate_output("Opening settings.")
    assert signing_output.language == "asl"
    assert signing_output.text == "Opening settings."
