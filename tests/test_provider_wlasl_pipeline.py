from dataclasses import dataclass

from unison_io_sign.providers.asl import ASLProvider
from unison_io_sign.keypoints import KeypointResult
from unison_io_sign.schemas import VideoSegment


@dataclass
class FakeExtractor:
    calls: int = 0

    def extract(self, frames):
        self.calls += 1
        return KeypointResult(hand_landmarks=["h1"], body_landmarks=["b1"])


@dataclass
class FakeClassifier:
    loaded: bool = True

    def predict(self, keypoints, hint_text=None):
        assert keypoints.hand_landmarks == ["h1"]
        text = hint_text or "from_model"
        gloss = ["OPEN", "SETTINGS"] if text else []
        confidence = 0.9
        return text, confidence, gloss


def test_asl_provider_with_injected_model_and_extractor():
    extractor = FakeExtractor()
    classifier = FakeClassifier()
    provider = ASLProvider(extractor=extractor, classifier=classifier)
    segment = VideoSegment(frames=["frame"], metadata={"text_hint": "open settings"})
    interp = provider.interpret_segment(segment)
    assert extractor.calls == 1
    assert interp.text == "open settings"
    assert interp.confidence == 0.9
    assert interp.raw_gloss == ["OPEN", "SETTINGS"]
