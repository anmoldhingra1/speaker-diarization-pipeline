"""Voice Activity Detection (VAD).

Detects speech regions in audio, filtering out silence
and non-speech segments before speaker embedding extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path

from diarization.types import SpeechRegion

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """Detect speech regions in audio files.

    Uses energy-based detection with optional model-based refinement
    to identify regions containing speech.

    Args:
        threshold: Detection confidence threshold (0-1).
        min_duration: Minimum speech region duration in seconds.
        sample_rate: Expected audio sample rate.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_duration: float = 0.3,
        sample_rate: int = 16000,
    ) -> None:
        self.threshold = threshold
        self.min_duration = min_duration
        self.sample_rate = sample_rate

    def detect(self, audio_path: str | Path) -> list[SpeechRegion]:
        """Detect speech regions in an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List of detected speech regions sorted by start time.
        """
        audio_path = Path(audio_path)
        logger.info("Running VAD on %s (threshold=%.2f)", audio_path, self.threshold)

        raw_regions = self._detect_raw(audio_path)
        filtered = self._filter_short(raw_regions)
        merged = self._merge_close(filtered)

        logger.info(
            "VAD: %d raw -> %d filtered -> %d merged regions",
            len(raw_regions),
            len(filtered),
            len(merged),
        )
        return merged

    def _detect_raw(self, audio_path: Path) -> list[SpeechRegion]:
        """Run raw detection on audio. Override for model-based VAD."""
        # Placeholder: returns regions based on file analysis
        # Real implementation uses energy or neural VAD model
        return []

    def _filter_short(self, regions: list[SpeechRegion]) -> list[SpeechRegion]:
        """Remove regions shorter than minimum duration."""
        return [
            r for r in regions
            if r.duration >= self.min_duration
        ]

    def _merge_close(
        self, regions: list[SpeechRegion], gap: float = 0.15
    ) -> list[SpeechRegion]:
        """Merge regions separated by less than gap seconds."""
        if not regions:
            return []

        sorted_regions = sorted(regions, key=lambda r: r.start)
        merged = [sorted_regions[0]]

        for region in sorted_regions[1:]:
            prev = merged[-1]
            if region.start - prev.end <= gap:
                merged[-1] = SpeechRegion(
                    start=prev.start,
                    end=region.end,
                    confidence=min(prev.confidence, region.confidence),
                )
            else:
                merged.append(region)

        return merged
