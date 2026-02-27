"""Tests for diarization.segmenter module."""

from __future__ import annotations

import pytest

from diarization.segmenter import SpeakerSegmenter
from diarization.types import SpeechRegion, Segment


class TestSpeakerSegmenter:
    """Tests for SpeakerSegmenter class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        segmenter = SpeakerSegmenter()
        assert segmenter.collar == 0.25

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        segmenter = SpeakerSegmenter(collar=0.5)
        assert segmenter.collar == 0.5

    def test_assign_labels_two_speakers(self) -> None:
        """Test label assignment for 2 speakers."""
        segmenter = SpeakerSegmenter()
        labels = segmenter._assign_labels([0, 1], num_speakers=2)
        assert labels[0] == "agent"
        assert labels[1] == "customer"

    def test_assign_labels_multi_speakers(self) -> None:
        """Test label assignment for multiple speakers."""
        segmenter = SpeakerSegmenter()
        labels = segmenter._assign_labels([0, 1, 2, 3], num_speakers=4)
        assert labels[0] == "speaker_0"
        assert labels[1] == "speaker_1"
        assert labels[2] == "speaker_2"
        assert labels[3] == "speaker_3"

    def test_assign_labels_single_speaker(self) -> None:
        """Test label assignment for single speaker."""
        segmenter = SpeakerSegmenter()
        labels = segmenter._assign_labels([0], num_speakers=1)
        assert labels[0] == "speaker_0"

    def test_merge_adjacent_empty(self) -> None:
        """Test merge with empty segments."""
        segmenter = SpeakerSegmenter()
        result = segmenter._merge_adjacent([])
        assert result == []

    def test_merge_adjacent_single(self) -> None:
        """Test merge with single segment."""
        segmenter = SpeakerSegmenter()
        seg = Segment(speaker="agent", start=0.0, end=1.0)
        result = segmenter._merge_adjacent([seg])
        assert len(result) == 1
        assert result[0].speaker == "agent"

    def test_merge_adjacent_same_speaker(self) -> None:
        """Test merge adjacent same-speaker segments."""
        segmenter = SpeakerSegmenter(collar=0.25)
        segments = [
            Segment(speaker="agent", start=0.0, end=1.0),
            Segment(speaker="agent", start=1.1, end=2.0),
        ]
        result = segmenter._merge_adjacent(segments)
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 2.0

    def test_merge_adjacent_different_speaker(self) -> None:
        """Test merge does not merge different speakers."""
        segmenter = SpeakerSegmenter(collar=0.25)
        segments = [
            Segment(speaker="agent", start=0.0, end=1.0),
            Segment(speaker="customer", start=1.1, end=2.0),
        ]
        result = segmenter._merge_adjacent(segments)
        assert len(result) == 2
        assert result[0].speaker == "agent"
        assert result[1].speaker == "customer"

    def test_merge_adjacent_large_gap(self) -> None:
        """Test merge respects gap threshold."""
        segmenter = SpeakerSegmenter(collar=0.25)
        segments = [
            Segment(speaker="agent", start=0.0, end=1.0),
            Segment(speaker="agent", start=2.0, end=3.0),  # gap > 0.25
        ]
        result = segmenter._merge_adjacent(segments)
        assert len(result) == 2

    def test_merge_adjacent_sorting(self) -> None:
        """Test merge sorts segments."""
        segmenter = SpeakerSegmenter(collar=0.5)
        segments = [
            Segment(speaker="agent", start=2.0, end=3.0),
            Segment(speaker="agent", start=0.0, end=1.0),
        ]
        result = segmenter._merge_adjacent(segments)
        assert result[0].start == 0.0
        assert result[-1].end == 3.0

    def test_merge_adjacent_confidence(self) -> None:
        """Test merge uses minimum confidence."""
        segmenter = SpeakerSegmenter(collar=0.5)
        segments = [
            Segment(speaker="agent", start=0.0, end=1.0, confidence=0.9),
            Segment(speaker="agent", start=1.2, end=2.0, confidence=0.7),
        ]
        result = segmenter._merge_adjacent(segments)
        assert len(result) == 1
        assert result[0].confidence == 0.7

    def test_build_segments_mismatch(self) -> None:
        """Test build_segments with mismatched lengths."""
        segmenter = SpeakerSegmenter()
        regions = [SpeechRegion(start=0.0, end=1.0)]
        labels = [0, 1]  # mismatch
        with pytest.raises(ValueError):
            segmenter.build_segments(regions, labels, 2)

    def test_build_segments_basic(self) -> None:
        """Test basic segment building."""
        segmenter = SpeakerSegmenter()
        regions = [
            SpeechRegion(start=0.0, end=1.0),
            SpeechRegion(start=1.0, end=2.0),
        ]
        labels = [0, 1]
        result = segmenter.build_segments(regions, labels, 2)
        assert len(result) == 2
        assert result[0].speaker == "agent"
        assert result[1].speaker == "customer"

    def test_build_segments_merges(self) -> None:
        """Test build_segments applies merging."""
        segmenter = SpeakerSegmenter(collar=0.5)
        regions = [
            SpeechRegion(start=0.0, end=1.0),
            SpeechRegion(start=1.2, end=2.0),
        ]
        labels = [0, 0]  # same speaker
        result = segmenter.build_segments(regions, labels, 1)
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 2.0

    def test_build_segments_many(self) -> None:
        """Test build_segments with many regions."""
        segmenter = SpeakerSegmenter()
        regions = [SpeechRegion(start=float(i), end=float(i + 1)) for i in range(20)]
        labels = [i % 2 for i in range(20)]
        result = segmenter.build_segments(regions, labels, 2)
        assert len(result) > 0
        assert all(s.speaker in ["agent", "customer"] for s in result)
