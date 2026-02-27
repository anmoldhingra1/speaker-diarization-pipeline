"""Tests for diarization.types module."""

from __future__ import annotations


from diarization.types import (
    DiarizationResult,
    PipelineConfig,
    Segment,
    SpeechRegion,
)


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.embedding_model == "emb_voxceleb"
        assert config.vad_threshold == 0.5
        assert config.min_segment_duration == 0.3
        assert config.collar == 0.25
        assert config.max_speakers == 10
        assert config.min_speakers == 2
        assert config.sample_rate == 16000

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = PipelineConfig(
            embedding_model="custom_model",
            vad_threshold=0.7,
            max_speakers=5,
        )
        assert config.embedding_model == "custom_model"
        assert config.vad_threshold == 0.7
        assert config.max_speakers == 5
        assert config.min_speakers == 2  # default

    def test_config_bounds(self) -> None:
        """Test config with boundary values."""
        config = PipelineConfig(
            vad_threshold=0.0,
            collar=0.0,
            max_speakers=1,
        )
        assert config.vad_threshold == 0.0
        assert config.collar == 0.0
        assert config.max_speakers == 1


class TestSpeechRegion:
    """Tests for SpeechRegion dataclass."""

    def test_basic_region(self) -> None:
        """Test basic speech region creation."""
        region = SpeechRegion(start=1.0, end=2.5)
        assert region.start == 1.0
        assert region.end == 2.5
        assert region.confidence == 1.0

    def test_region_duration(self) -> None:
        """Test duration calculation."""
        region = SpeechRegion(start=0.5, end=3.5, confidence=0.9)
        assert region.duration == 3.0
        assert region.confidence == 0.9

    def test_zero_duration_region(self) -> None:
        """Test region with zero duration."""
        region = SpeechRegion(start=1.0, end=1.0)
        assert region.duration == 0.0

    def test_negative_duration_region(self) -> None:
        """Test region with negative duration."""
        region = SpeechRegion(start=2.0, end=1.0)
        assert region.duration == -1.0

    def test_region_confidence_bounds(self) -> None:
        """Test region with confidence values."""
        region_high = SpeechRegion(start=0.0, end=1.0, confidence=0.99)
        region_low = SpeechRegion(start=0.0, end=1.0, confidence=0.01)
        assert region_high.confidence == 0.99
        assert region_low.confidence == 0.01


class TestSegment:
    """Tests for Segment dataclass."""

    def test_basic_segment(self) -> None:
        """Test basic segment creation."""
        seg = Segment(speaker="agent", start=0.0, end=1.5)
        assert seg.speaker == "agent"
        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.confidence == 0.0

    def test_segment_with_confidence(self) -> None:
        """Test segment with confidence."""
        seg = Segment(
            speaker="customer",
            start=1.5,
            end=3.0,
            confidence=0.95,
        )
        assert seg.speaker == "customer"
        assert seg.duration == 1.5
        assert seg.confidence == 0.95

    def test_segment_duration(self) -> None:
        """Test segment duration property."""
        seg = Segment(speaker="agent", start=10.0, end=15.0)
        assert seg.duration == 5.0

    def test_segment_repr(self) -> None:
        """Test segment string representation."""
        seg = Segment(speaker="agent", start=0.5, end=2.3, confidence=0.87)
        repr_str = repr(seg)
        assert "Segment" in repr_str
        assert "agent" in repr_str
        assert "0.50" in repr_str
        assert "2.30" in repr_str
        assert "0.870" in repr_str

    def test_different_speaker_labels(self) -> None:
        """Test segments with different speaker labels."""
        labels = ["agent", "customer", "speaker_0", "speaker_1"]
        for label in labels:
            seg = Segment(speaker=label, start=0.0, end=1.0)
            assert seg.speaker == label


class TestDiarizationResult:
    """Tests for DiarizationResult dataclass."""

    def test_empty_result(self) -> None:
        """Test empty diarization result."""
        result = DiarizationResult(segments=[], num_speakers=0)
        assert result.segments == []
        assert result.num_speakers == 0
        assert result.total_duration == 0.0

    def test_result_with_segments(self) -> None:
        """Test result with multiple segments."""
        segments = [
            Segment(speaker="agent", start=0.0, end=2.0),
            Segment(speaker="customer", start=2.0, end=5.0),
            Segment(speaker="agent", start=5.0, end=7.0),
        ]
        result = DiarizationResult(segments=segments, num_speakers=2)
        assert len(result.segments) == 3
        assert result.num_speakers == 2
        assert result.total_duration == 7.0

    def test_total_duration(self) -> None:
        """Test total duration calculation."""
        segments = [
            Segment(speaker="agent", start=0.0, end=2.0),
            Segment(speaker="customer", start=2.5, end=5.5),
        ]
        result = DiarizationResult(segments=segments, num_speakers=2)
        assert result.total_duration == 5.5

    def test_speakers(self) -> None:
        """Test speakers() method."""
        segments = [
            Segment(speaker="customer", start=0.0, end=1.0),
            Segment(speaker="agent", start=1.0, end=2.0),
            Segment(speaker="customer", start=2.0, end=3.0),
        ]
        result = DiarizationResult(segments=segments, num_speakers=2)
        speakers = result.speakers()
        assert speakers == ["agent", "customer"]

    def test_speakers_empty(self) -> None:
        """Test speakers() with no segments."""
        result = DiarizationResult(segments=[], num_speakers=0)
        assert result.speakers() == []

    def test_segments_for(self) -> None:
        """Test segments_for() method."""
        segments = [
            Segment(speaker="agent", start=0.0, end=1.0),
            Segment(speaker="customer", start=1.0, end=2.0),
            Segment(speaker="agent", start=2.0, end=3.0),
        ]
        result = DiarizationResult(segments=segments, num_speakers=2)

        agent_segs = result.segments_for("agent")
        assert len(agent_segs) == 2
        assert all(s.speaker == "agent" for s in agent_segs)

        customer_segs = result.segments_for("customer")
        assert len(customer_segs) == 1
        assert customer_segs[0].speaker == "customer"

    def test_segments_for_nonexistent(self) -> None:
        """Test segments_for() with nonexistent speaker."""
        segments = [Segment(speaker="agent", start=0.0, end=1.0)]
        result = DiarizationResult(segments=segments, num_speakers=1)
        assert result.segments_for("nonexistent") == []

    def test_to_rttm(self) -> None:
        """Test RTTM format export."""
        result = DiarizationResult(
            segments=[
                Segment(speaker="agent", start=0.0, end=2.0),
                Segment(speaker="customer", start=2.0, end=5.0),
            ],
            num_speakers=2,
            audio_path="test.wav",
        )
        rttm = result.to_rttm()
        assert "SPEAKER" in rttm
        assert "test.wav" in rttm
        assert "agent" in rttm
        assert "customer" in rttm

    def test_result_with_audio_path(self) -> None:
        """Test result with audio path set."""
        segments = [Segment(speaker="agent", start=0.0, end=1.0)]
        result = DiarizationResult(
            segments=segments,
            num_speakers=1,
            audio_path="/path/to/audio.wav",
        )
        assert result.audio_path == "/path/to/audio.wav"
