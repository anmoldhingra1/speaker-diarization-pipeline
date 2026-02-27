"""Speaker diarization pipeline for audio processing.

Segments and labels agent vs. customer speech from raw audio input
with precise timestamp boundaries.
"""

from diarization.pipeline import DiarizationPipeline
from diarization.segmenter import SpeakerSegmenter
from diarization.embeddings import EmbeddingExtractor

__version__ = "0.1.0"
__all__ = ["DiarizationPipeline", "SpeakerSegmenter", "EmbeddingExtractor"]
