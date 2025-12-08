# unison-io-sign

Sign language I/O services for UnisonOS (ASL-first). This repo hosts the sign presence detector, interpreter pipeline, provider abstraction, avatar output adapter, and shared schemas used by the sign modality.

## Status
Phase 0 scaffolding — schemas, provider interface, ASL provider stub, and tests. No runtime server yet.

## Layout
- `src/unison_io_sign/schemas.py` — shared dataclasses for presence, interpretation, signing output.
- `src/unison_io_sign/provider.py` — `SignLanguageProvider` protocol and provider registry helper.
- `src/unison_io_sign/providers/asl.py` — ASL provider stub implementing the protocol (with optional model path hook).
- `src/unison_io_sign/detector.py` — lightweight presence detector skeleton.
- `src/unison_io_sign/interpreter.py` — segmentation + provider wiring skeleton.
- `tests/` — unit tests for schema serialization and provider contracts.
- `docs/MODEL_SETUP.md` — plan for wiring MediaPipe + WLASL model locally.
- `docs/REAL_MODEL_TODO.md` — drop-in guide for plugging a real ASL ONNX model + labels.

Planned additions:
- `unison_io_sign_detector` — lightweight sign presence detection.
- `unison_io_sign_interpreter` — segmentation/preprocessing pipeline.
- `unison_io_sign_avatar` — signing avatar backend abstraction.

## Quickstart (dev)
```bash
cd unison-io-sign
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

## Roadmap (excerpt)
- Phase 0: schemas, provider protocol, ASL stub, tests. ✅
- Phase 1: presence detector + interpreter skeleton + integration tests. ✅ (current)
- Phase 2: ASL provider MVP and replay tests. ✅ (current stub uses text_hint or optional UNISON_ASL_MODEL_PATH placeholder; real model next)
- Phase 3: avatar output adapter + shell hooks.
- Phase 4: policy/consent integration and preferences.
- Phase 5: perf tuning and multi-language readiness.
