"""
Provider abstraction for sign-language interpretation and signing output.
"""

from __future__ import annotations

from typing import Dict, Protocol

from .schemas import SignInterpretation, SigningOutput, VideoSegment


class SignLanguageProvider(Protocol):
    @property
    def language_code(self) -> str:  # e.g., "asl", "bsl"
        ...

    def interpret_segment(self, segment: VideoSegment) -> SignInterpretation:
        """
        Convert a video/keypoint segment into a SignInterpretation.
        """

    def generate_output(self, text: str) -> SigningOutput:
        """
        Convert natural-language text into signing output for this language.
        """


_PROVIDERS: Dict[str, SignLanguageProvider] = {}


def register_provider(provider: SignLanguageProvider) -> None:
    _PROVIDERS[provider.language_code] = provider


def get_provider(language_code: str) -> SignLanguageProvider:
    if language_code not in _PROVIDERS:
        raise KeyError(f"No provider registered for language: {language_code}")
    return _PROVIDERS[language_code]
