from __future__ import annotations

"""
Keypoint extraction utilities.

This module is designed to prefer MediaPipe for hand/body landmarks, with a graceful
fallback to a no-op extractor when dependencies are unavailable.
"""

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional


@dataclass
class KeypointResult:
    hand_landmarks: List[Any]  # structure depends on backend; kept opaque here
    body_landmarks: List[Any]
    frame_features: List[List[float]] = field(default_factory=list)


class _NoOpExtractor:
    def extract(self, frames: List[Any]) -> KeypointResult:
        return KeypointResult(hand_landmarks=[], body_landmarks=[], frame_features=[])


class MediaPipeExtractor:
    """
    Thin wrapper to avoid hard dependency failures when mediapipe is absent.
    """

    def __init__(self):
        try:
            import mediapipe as mp  # type: ignore
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(f"mediapipe not available: {exc}") from exc
        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frames: List[Any]) -> KeypointResult:
        # Note: frames are assumed to be RGB images (numpy arrays). In Phase 2,
        # tests use empty frames; this path will be used once real frames are passed.
        hand_landmarks = []
        body_landmarks = []
        frame_features: List[List[float]] = []

        def _flatten_landmarks(lms) -> List[float]:
            flat: List[float] = []
            if lms:
                for lm in lms:
                    for pt in lm.landmark:
                        flat.extend([float(pt.x), float(pt.y), float(pt.z)])
            return flat

        for frame in frames:
            hand_res = self._hands.process(frame)
            pose_res = self._pose.process(frame)
            if hand_res.multi_hand_landmarks:
                hand_landmarks.extend(hand_res.multi_hand_landmarks)
            if pose_res.pose_landmarks:
                body_landmarks.append(pose_res.pose_landmarks)
            hands_flat = _flatten_landmarks(hand_res.multi_hand_landmarks) if hand_res else []
            body_flat = _flatten_landmarks([pose_res.pose_landmarks]) if pose_res and pose_res.pose_landmarks else []
            frame_features.append(hands_flat + body_flat)

        return KeypointResult(hand_landmarks=hand_landmarks, body_landmarks=body_landmarks, frame_features=frame_features)


def make_extractor(backend: Optional[Literal["mediapipe"]] = "mediapipe"):
    if backend == "mediapipe":
        try:
            return MediaPipeExtractor()
        except Exception:
            return _NoOpExtractor()
    return _NoOpExtractor()
