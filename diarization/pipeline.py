"""Core diarization pipeline.

Orchestrates VAD, embedding extraction, and clustering
to produce speaker-labeled segments from raw audio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from diarization.segmenter import SpeakerSegmenter
from diarization.embeddings import EmbeddingExtractor
from diarization.clustering import SpectralClusterer
from diarization.vad import VoiceActivityDetector
from diarization.types import DiarizationResult, Segment, PipelineConfig

logger = logging.getLogger(__name__)


class DiarizationPipeline:
    """End-to-end speaker diarization pipeline.

    Processes raw audio through voice activity detection,
    speaker embedding extraction, and spectral clustering
    to produce time-stamped, speaker-labeled segments.

    Args:
        config: Pipeline configuration. Uses defaults if not provided.
        model_dir: Directory containing pretrained model weights.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        model_dir: Optional[Path] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.model_dir = model_dir or Path("models")

        self._vad = VoiceActivityDetector(
            threshold=self.config.vad_threshold,
            min_duration=self.config.min_segment_duration,
        )
        self._embedder = EmbeddingExtractor(
            model_path=self.model_dir / self.config.embedding_model,
        )
        self._clusterer = SpectralClusterer(
            max_speakers=self.config.max_speakers,
            min_speakers=self.config.min_speakers,
        )
        self._segmenter = SpeakerSegmenter(
            collar=self.config.collar,
        )

        self._is_initialized = False
        logger.info("Pipeline created with config: %s", self.config)

    def initialize(self) -> None:
        """Load models into memory. Called automatically on first process()."""
        if self._is_initialized:
            return
        logger.info("Loading models from %s", self.model_dir)
        self._embedder.load()
        self._is_initialized = True
        logger.info("Pipeline initialized")

    def process(self, audio_path: str | Path) -> DiarizationResult:
        """Run diarization on an audio file.

        Args:
            audio_path: Path to the audio file (WAV, MP3, FLAC).

        Returns:
            DiarizationResult with speaker-labeled segments.

        Raises:
            FileNotFoundError: If audio file does not exist.
            ValueError: If audio format is unsupported.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self.initialize()
        logger.info("Processing: %s", audio_path)

        # Step 1: Voice activity detection
        speech_regions = self._vad.detect(audio_path)
        logger.info("Found %d speech regions", len(speech_regions))

        if not speech_regions:
            return DiarizationResult(segments=[], num_speakers=0)

        # Step 2: Extract speaker embeddings per region
        embeddings = self._embedder.extract(audio_path, speech_regions)
        logger.info("Extracted %d embeddings", len(embeddings))

        # Step 3: Cluster embeddings into speaker groups
        labels = self._clusterer.cluster(embeddings)
        num_speakers = len(set(labels))
        logger.info("Identified %d speakers", num_speakers)

        # Step 4: Build speaker-labeled segments
        segments = self._segmenter.build_segments(
            speech_regions, labels, num_speakers
        )

        return DiarizationResult(
            segments=segments,
            num_speakers=num_speakers,
            audio_path=str(audio_path),
        )

    def process_batch(
        self, audio_paths: list[str | Path]
    ) -> list[DiarizationResult]:
        """Process multiple audio files.

        Args:
            audio_paths: List of paths to audio files.

        Returns:
            List of DiarizationResult, one per file.
        """
        self.initialize()
        results = []
        for i, path in enumerate(audio_paths, 1):
            logger.info("Processing file %d/%d: %s", i, len(audio_paths), path)
            results.append(self.process(path))
        return results
