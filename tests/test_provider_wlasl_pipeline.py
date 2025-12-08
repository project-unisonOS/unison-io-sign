from dataclasses import dataclass

from unison_io_sign.providers.asl import ASLProvider
from unison_io_sign.keypoints import KeypointResult
from unison_io_sign.schemas import VideoSegment
import numpy as np


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


class FakeONNXSession:
    def __init__(self):
        self._inputs = [type("I", (), {"name": "input"})]

    def get_inputs(self):
        return self._inputs

    def run(self, _, inputs):
        arr = inputs["input"]
        # return a score vector using counts
        score = float(np.sum(arr))
        return [np.array([score], dtype=np.float32)]


def test_wlasl_classifier_with_onnx_session(monkeypatch, tmp_path):
    fake_model = tmp_path / "model.onnx"
    fake_model.write_text("stub")
    monkeypatch.setenv("UNISON_SIGN_MODEL_PATH_ASL", str(fake_model))

    # inject fake session into classifier via provider creation
    extractor = FakeExtractor()
    session = FakeONNXSession()

    # patch WLASLClassifier to return our fake session
    from unison_io_sign import wlasl_classifier

    orig_init = wlasl_classifier.WLASLClassifier.__init__

    def _init(self, model_path, session=None):
        orig_init(self, model_path, session=session or FakeONNXSession())

    monkeypatch.setattr(wlasl_classifier.WLASLClassifier, "__init__", _init)

    provider = ASLProvider(extractor=extractor)
    segment = VideoSegment(frames=["frame"], metadata={})
    interp = provider.interpret_segment(segment)
    assert interp.text == "asl_wlasl_onnx" or interp.text == ""
    assert interp.confidence >= 0.6
