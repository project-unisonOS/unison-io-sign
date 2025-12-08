from unison_io_sign.providers.asl import ASLProvider
from unison_io_sign.schemas import VideoSegment


def test_asl_provider_uses_text_hint():
    provider = ASLProvider()
    segment = VideoSegment(metadata={"text_hint": "open settings"})
    interp = provider.interpret_segment(segment)
    assert interp.text == "open settings"
    assert interp.confidence == 0.75


def test_asl_provider_model_path_without_file_falls_back(monkeypatch, tmp_path):
    fake_model = tmp_path / "model.pt"
    # deliberately do not create the file; loader should fail and fall back
    monkeypatch.setenv("UNISON_SIGN_MODEL_PATH_ASL", str(fake_model))
    provider = ASLProvider()
    segment = VideoSegment()
    interp = provider.interpret_segment(segment)
    # Falls back to low-confidence path
    assert interp.confidence == 0.2


def test_asl_provider_model_path_with_file_uses_model(tmp_path, monkeypatch):
    # create a fake model file so loader marks as available
    fake_model = tmp_path / "model.pt"
    fake_model.write_text("stub")
    monkeypatch.setenv("UNISON_SIGN_MODEL_PATH_ASL", str(fake_model))
    # Inject a fake classifier that reports loaded and returns high confidence
    class FakeLoadedClassifier:
        loaded = True

        def predict(self, keypoints, hint_text=None):
            return hint_text or "from_model", 0.9, ["OPEN"]

    provider = ASLProvider(classifier=FakeLoadedClassifier())
    segment = VideoSegment()
    interp = provider.interpret_segment(segment)
    # With model loaded, confidence is elevated
    assert interp.confidence >= 0.6
