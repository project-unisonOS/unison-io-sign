from dataclasses import dataclass

from unison_io_sign.providers.asl import ASLProvider
from unison_io_sign.keypoints import KeypointResult
from unison_io_sign.schemas import VideoSegment
import numpy as np
import onnx
from onnx import helper, TensorProto


@dataclass
class FakeExtractor:
    calls: int = 0

    def extract(self, frames):
        self.calls += 1
        return KeypointResult(
            hand_landmarks=[(0.1, 0.2, 0.3)],
            body_landmarks=[(0.4, 0.5, 0.6)],
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


def _build_tiny_onnx_model(path):
    # Model: ReduceSum over axis=1 to produce a single score.
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", "features"])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch"])
    axes_initializer = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    node = helper.make_node("ReduceSum", inputs=["input", "axes"], outputs=["output"], keepdims=0)
    graph = helper.make_graph([node], "sum_model", [input_tensor], [output_tensor], initializer=[axes_initializer])
    model = helper.make_model(graph, producer_name="unison-io-sign-test", opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.save(model, path)


def test_wlasl_classifier_end_to_end_with_real_onnx(tmp_path, monkeypatch):
    model_path = tmp_path / "tiny.onnx"
    _build_tiny_onnx_model(model_path)
    monkeypatch.setenv("UNISON_SIGN_MODEL_PATH_ASL", str(model_path))

    # Fake extractor that emits coordinates
    @dataclass
    class Point:
        x: float
        y: float
        z: float

    @dataclass
    class Extractor:
        def extract(self, frames):
            return KeypointResult(
                hand_landmarks=[Point(0.1, 0.2, 0.3), Point(0.4, 0.5, 0.6)],
                body_landmarks=[Point(0.7, 0.8, 0.9)],
            )

    import onnxruntime as ort
    from unison_io_sign.wlasl_classifier import WLASLClassifier

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    provider = ASLProvider(extractor=Extractor(), classifier=WLASLClassifier(str(model_path), session=session))
    segment = VideoSegment(frames=["frame"], metadata={})
    interp = provider.interpret_segment(segment)
    # With actual ONNX run, we get the default text and a confidence derived from sum of coords.
    assert interp.text in {"asl_wlasl_onnx", ""}
    assert interp.confidence >= 0.6
