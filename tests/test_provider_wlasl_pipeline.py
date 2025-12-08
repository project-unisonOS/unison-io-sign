from dataclasses import dataclass
import json
from pathlib import Path

from unison_io_sign.providers.asl import ASLProvider
from unison_io_sign.keypoints import KeypointResult
from unison_io_sign.schemas import VideoSegment
import numpy as np
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "asl"


@dataclass
class FakeExtractor:
    calls: int = 0

    def extract(self, frames):
        self.calls += 1
        return KeypointResult(
            hand_landmarks=[(0.1, 0.2, 0.3)],
            body_landmarks=[(0.4, 0.5, 0.6)],
            frame_features=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
        )


@dataclass
class FakeClassifier:
    loaded: bool = True

    def predict(self, keypoints, hint_text=None):
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


def test_wlasl_classifier_with_fixture(monkeypatch):
    model_path = FIXTURES / "wlasl_stub.onnx"
    labels_path = FIXTURES / "wlasl_labels.json"
    keypoints_path = FIXTURES / "keypoints_open_settings.json"

    monkeypatch.setenv("UNISON_SIGN_MODEL_PATH_ASL", str(model_path))
    monkeypatch.setenv("UNISON_SIGN_LABELS_PATH_ASL", str(labels_path))

    data = json.loads(keypoints_path.read_text())
    frames = data["frames"]

    @dataclass
    class Extractor:
        def extract(self, frames_in):
            return KeypointResult(hand_landmarks=[], body_landmarks=[], frame_features=frames)

    provider = ASLProvider(extractor=Extractor())
    segment = VideoSegment(frames=["frame"], metadata={})
    interp = provider.interpret_segment(segment)
    # Expect class id 1 from the stub logits (0.1, 0.9)
    assert interp.text == "open browser"
    assert interp.raw_gloss == ["OPEN", "BROWSER"]
    assert interp.confidence >= 0.6
