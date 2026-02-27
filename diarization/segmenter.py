"""Speaker segmentation and segment refinement."""

from __future__ import annotations

import logging
from typing import Sequence

from diarization.types import SpeechRegion, Segment

logger = logging.getLogger(__name__)


class SpeakerSegmenter:
    """Builds and refines speaker-labeled segments from clustering output.

    Applies collar-based merging to reduce over-segmentation and
    assigns human-readable speaker labels (agent, customer, speaker_N).

    Args:
        collar: Merge segments from the same speaker if gap is below
                this threshold (in seconds).
    """

    def __init__(self, collar: float = 0.25) -> None:
        self.collar = collar

    def build_segments(
        self,
        regions: Sequence[SpeechRegion],
        labels: Sequence[int],
        num_speakers: int,
    ) -> list[Segment]:
        """Build speaker segments from regions and cluster labels.

        Args:
            regions: Detected speech regions.
            labels: Cluster label for each region.
            num_speakers: Total number of speakers detected.

        Returns:
            Merged, sorted list of speaker segments.
        """
        if len(regions) != len(labels):
            raise ValueError(
                f"Mismatch: {len(regions)} regions vs {len(labels)} labels"
            )

        speaker_names = self._assign_labels(labels, num_speakers)

        raw_segments = [
            Segment(
                speaker=speaker_names[label],
                start=region.start,
                end=region.end,
                confidence=region.confidence,
            )
            for region, label in zip(regions, labels)
        ]

        merged = self._merge_adjacent(raw_segments)
        logger.info(
            "Built %d segments from %d regions (merged %d)",
            len(merged),
            len(raw_segments),
            len(raw_segments) - len(merged),
        )
        return merged

    def _assign_labels(
        self, labels: Sequence[int], num_speakers: int
    ) -> dict[int, str]:
        """Map cluster IDs to human-readable labels."""
        if num_speakers == 2:
            return {0: "agent", 1: "customer"}

        return {
            i: f"speaker_{i}" for i in range(num_speakers)
        }

    def _merge_adjacent(self, segments: list[Segment]) -> list[Segment]:
        """Merge consecutive segments from the same speaker within collar."""
        if not segments:
            return []

        sorted_segs = sorted(segments, key=lambda s: s.start)
        merged: list[Segment] = [sorted_segs[0]]

        for seg in sorted_segs[1:]:
            prev = merged[-1]
            gap = seg.start - prev.end

            if seg.speaker == prev.speaker and gap <= self.collar:
                merged[-1] = Segment(
                    speaker=prev.speaker,
                    start=prev.start,
                    end=seg.end,
                    confidence=min(prev.confidence, seg.confidence),
                )
            else:
                merged.append(seg)

        return merged
