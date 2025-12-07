from unison_io_sign.schemas import (
    AvatarInstructions,
    SignInterpretation,
    SignPresenceEvent,
    SigningOutput,
    VideoSegment,
)


def test_signing_output_to_dict_round_trip():
    output = SigningOutput(language="asl", text="Hello", gloss=["HELLO"])
    data = output.to_dict()
    assert data["language"] == "asl"
    assert data["text"] == "Hello"
    assert data["gloss"] == ["HELLO"]
    assert data["avatar_instructions"]["version"] == "1.0"


def test_presence_event_serializes():
    event = SignPresenceEvent(
        event_type="sign_presence_detected",
        timestamp="2025-01-01T00:00:00Z",
        source="unison-io-sign-detector",
        confidence=0.9,
        language_hint="asl",
    )
    data = event.to_dict()
    assert data["event_type"] == "sign_presence_detected"
    assert data["confidence"] == 0.9
    assert data["language_hint"] == "asl"


def test_interpretation_stub_factory():
    segment = VideoSegment()
    interp = SignInterpretation.from_stub(
        language="asl", text="open settings", intent={"name": "open_app"}, gloss=["OPEN", "SETTINGS"], segment=segment
    )
    assert interp.language == "asl"
    assert interp.intent == {"name": "open_app"}
    assert interp.raw_gloss == ["OPEN", "SETTINGS"]
    assert interp.segment_id == segment.segment_id
