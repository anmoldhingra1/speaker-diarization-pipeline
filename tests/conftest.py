"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_config():
    """Provide a sample pipeline configuration."""
    from diarization.types import PipelineConfig
    return PipelineConfig()


@pytest.fixture
def sample_speech_regions():
    """Provide sample speech regions."""
    from diarization.types import SpeechRegion
    return [
        SpeechRegion(start=0.0, end=1.0, confidence=0.9),
        SpeechRegion(start=1.5, end=2.5, confidence=0.85),
        SpeechRegion(start=3.0, end=4.5, confidence=0.92),
    ]


@pytest.fixture
def sample_segments():
    """Provide sample speaker segments."""
    from diarization.types import Segment
    return [
        Segment(speaker="agent", start=0.0, end=1.0, confidence=0.9),
        Segment(speaker="customer", start=1.5, end=2.5, confidence=0.85),
        Segment(speaker="agent", start=3.0, end=4.5, confidence=0.92),
    ]
