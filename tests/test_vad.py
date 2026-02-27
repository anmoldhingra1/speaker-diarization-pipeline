"""Tests for diarization.vad module."""

from __future__ import annotations

from pathlib import Path

from diarization.vad import VoiceActivityDetector
from diarization.types import SpeechRegion


class TestVoiceActivityDetector:
    """Tests for VoiceActivityDetector class."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        vad = VoiceActivityDetector()
        assert vad.threshold == 0.5
        assert vad.min_duration == 0.3
        assert vad.sample_rate == 16000

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        vad = VoiceActivityDetector(
            threshold=0.7,
            min_duration=0.5,
            sample_rate=8000,
        )
        assert vad.threshold == 0.7
        assert vad.min_duration == 0.5
        assert vad.sample_rate == 8000

    def test_filter_short(self) -> None:
        """Test _filter_short removes short regions."""
        vad = VoiceActivityDetector(min_duration=0.5)
        regions = [
            SpeechRegion(start=0.0, end=0.2),  # too short
            SpeechRegion(start=1.0, end=1.6),  # ok
            SpeechRegion(start=2.0, end=2.3),  # too short
            SpeechRegion(start=3.0, end=4.0),  # ok
        ]
        filtered = vad._filter_short(regions)
        assert len(filtered) == 2
        assert filtered[0].duration == 0.6
        assert filtered[1].duration == 1.0

    def test_filter_short_empty(self) -> None:
        """Test _filter_short with empty list."""
        vad = VoiceActivityDetector()
        assert vad._filter_short([]) == []

    def test_filter_short_all_too_short(self) -> None:
        """Test _filter_short when all regions are too short."""
        vad = VoiceActivityDetector(min_duration=1.0)
        regions = [
            SpeechRegion(start=0.0, end=0.5),
            SpeechRegion(start=1.0, end=1.3),
        ]
        assert vad._filter_short(regions) == []

    def test_merge_close(self) -> None:
        """Test _merge_close merges nearby regions."""
        vad = VoiceActivityDetector()
        regions = [
            SpeechRegion(start=0.0, end=1.0),
            SpeechRegion(start=1.1, end=2.0),  # close enough
            SpeechRegion(start=3.0, end=4.0),  # gap > 0.15
        ]
        merged = vad._merge_close(regions, gap=0.15)
        assert len(merged) == 2
        assert merged[0].start == 0.0
        assert merged[0].end == 2.0
        assert merged[1].start == 3.0
        assert merged[1].end == 4.0

    def test_merge_close_custom_gap(self) -> None:
        """Test _merge_close with custom gap."""
        vad = VoiceActivityDetector()
        regions = [
            SpeechRegion(start=0.0, end=1.0),
            SpeechRegion(start=1.5, end=2.0),
        ]
        merged_small = vad._merge_close(regions, gap=0.3)
        assert len(merged_small) == 2

        merged_large = vad._merge_close(regions, gap=0.6)
        assert len(merged_large) == 1
        assert merged_large[0].end == 2.0

    def test_merge_close_empty(self) -> None:
        """Test _merge_close with empty list."""
        vad = VoiceActivityDetector()
        assert vad._merge_close([]) == []

    def test_merge_close_single(self) -> None:
        """Test _merge_close with single region."""
        vad = VoiceActivityDetector()
        regions = [SpeechRegion(start=0.0, end=1.0)]
        merged = vad._merge_close(regions)
        assert len(merged) == 1
        assert merged[0].start == 0.0
        assert merged[0].end == 1.0

    def test_merge_close_maintains_order(self) -> None:
        """Test _merge_close maintains sorted order."""
        vad = VoiceActivityDetector()
        regions = [
            SpeechRegion(start=5.0, end=6.0),
            SpeechRegion(start=1.0, end=2.0),
            SpeechRegion(start=3.0, end=4.0),
        ]
        merged = vad._merge_close(regions, gap=0.15)
        assert merged[0].start == 1.0
        assert merged[1].start == 3.0
        assert merged[2].start == 5.0

    def test_merge_close_confidence(self) -> None:
        """Test _merge_close uses minimum confidence."""
        vad = VoiceActivityDetector()
        regions = [
            SpeechRegion(start=0.0, end=1.0, confidence=0.9),
            SpeechRegion(start=1.05, end=2.0, confidence=0.7),
        ]
        merged = vad._merge_close(regions, gap=0.1)
        assert len(merged) == 1
        assert merged[0].confidence == 0.7

    def test_detect_raw(self) -> None:
        """Test _detect_raw returns empty list by default."""
        vad = VoiceActivityDetector()
        raw = vad._detect_raw(Path("dummy.wav"))
        assert raw == []


class TestVoiceActivityDetectorIntegration:
    """Integration tests for VoiceActivityDetector."""

    def test_detect_returns_sorted_regions(self) -> None:
        """Test detect returns sorted regions."""
        vad = VoiceActivityDetector()
        result = vad.detect("dummy.wav")
        if len(result) > 1:
            for i in range(len(result) - 1):
                assert result[i].start <= result[i + 1].start

    def test_detect_empty_file(self) -> None:
        """Test detect with file that produces no regions."""
        vad = VoiceActivityDetector()
        result = vad.detect("nonexistent.wav")
        assert isinstance(result, list)
        assert all(isinstance(r, SpeechRegion) for r in result)

    def test_detect_filters_and_merges(self) -> None:
        """Test detect applies filtering and merging."""
        vad = VoiceActivityDetector(min_duration=0.5)
        result = vad.detect("dummy.wav")
        # All returned regions should meet minimum duration
        assert all(r.duration >= 0.5 for r in result)
