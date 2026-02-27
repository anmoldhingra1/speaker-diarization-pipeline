"""Type definitions for the diarization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for the diarization pipeline."""

    embedding_model: str = "emb_voxceleb"
    vad_threshold: float = 0.5
    min_segment_duration: float = 0.3
    collar: float = 0.25
    max_speakers: int = 10
    min_speakers: int = 2
    sample_rate: int = 16000


@dataclass
class SpeechRegion:
    """A detected region of speech activity."""

    start: float
    end: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Segment:
    """A speaker-labeled audio segment."""

    speaker: str
    start: float
    end: float
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return (
            f"Segment(speaker={self.speaker!r}, "
            f"{self.start:.2f}s - {self.end:.2f}s, "
            f"conf={self.confidence:.3f})"
        )


@dataclass
class DiarizationResult:
    """Result of running the diarization pipeline on an audio file."""

    segments: list[Segment]
    num_speakers: int
    audio_path: Optional[str] = None

    @property
    def total_duration(self) -> float:
        if not self.segments:
            return 0.0
        return max(s.end for s in self.segments)

    def speakers(self) -> list[str]:
        """Return unique speaker labels."""
        return sorted(set(s.speaker for s in self.segments))

    def segments_for(self, speaker: str) -> list[Segment]:
        """Return segments for a specific speaker."""
        return [s for s in self.segments if s.speaker == speaker]

    def to_rttm(self) -> str:
        """Export in RTTM format for evaluation."""
        lines = []
        for seg in self.segments:
            duration = seg.end - seg.start
            lines.append(
                f"SPEAKER {self.audio_path or 'unknown'} 1 "
                f"{seg.start:.3f} {duration:.3f} "
                f"<NA> <NA> {seg.speaker} <NA> <NA>"
            )
        return "\n".join(lines)
