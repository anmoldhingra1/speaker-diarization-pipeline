[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/anmoldhingra1/speaker-diarization-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/anmoldhingra1/speaker-diarization-pipeline/actions)

# Speaker Diarization Pipeline

A speaker diarization pipeline for agent/customer audio. It segments and labels speech from raw audio input with timestamp boundaries, with a bias toward two-speaker conversational settings.

## Why This Exists

Voice applications need to know more than what was said. They need to know who spoke, when turns changed, and where overlapping speech or silence affects downstream interpretation.

This repo is a public slice of that audio-processing layer: voice activity detection, speaker embeddings, clustering, and segment refinement.

## Features

- **Automatic Speaker Segmentation**: "Who spoke when?" — precise temporal speaker identification
- **Agent-Customer Separation**: Optimized for 2-speaker contact center scenarios
- **Sub-Second Accuracy**: Timestamp boundaries accurate to ±100ms
- **Confidence Scoring**: Per-segment confidence values for quality filtering
- **Batch Processing**: Efficient concurrent processing of multiple files
- **Overlapping Speech Handling**: Explicit modeling of simultaneous speech
- **Audio Quality Robustness**: Tested on compressed VoIP, background noise, variable sample rates
- **Minimal Dependencies**: Clean, focused API with numpy as primary dependency

## Architecture

The pipeline orchestrates four core components:

```
Raw Audio
    ↓
Voice Activity Detection (VAD)
    ↓ [detects speech regions, filters silence]
Speaker Embedding Extraction
    ↓ [generates fixed-dim speaker-discriminative vectors]
Spectral Clustering
    ↓ [estimates speaker count, assigns speaker labels]
Segment Refinement
    ↓ [merges fragmented turns, applies temporal constraints]
Speaker-Labeled Segments [JSON output]
```

## Installation

```bash
git clone https://github.com/anmoldhingra1/speaker-diarization-pipeline.git
cd speaker-diarization-pipeline
pip install -e ".[dev]"
```

### Requirements

- Python 3.9 or higher
- numpy >= 1.24

## Quick Start

```python
from diarization import DiarizationPipeline

# Initialize pipeline
pipeline = DiarizationPipeline()

# Process audio file
result = pipeline.process("audio.wav")

# Inspect results
for segment in result.segments:
    print(f"{segment.speaker:8s} | {segment.start:7.2f}s - {segment.end:7.2f}s | confidence: {segment.confidence:.3f}")

# Example output:
# agent    |    0.23s -    2.31s | confidence: 0.957
# customer |    2.35s -    8.14s | confidence: 0.941
# agent    |    8.18s -   10.52s | confidence: 0.963
```

## Usage Examples

### Batch Processing

```python
from diarization import DiarizationPipeline
import glob

pipeline = DiarizationPipeline()
audio_files = glob.glob("call_recordings/*.wav")
results = pipeline.process_batch(audio_files)

for path, result in zip(audio_files, results):
    print(f"{path}: {result.num_speakers} speakers, {len(result.segments)} segments")
```

### Custom Configuration

```python
from diarization import DiarizationPipeline
from diarization.types import PipelineConfig

config = PipelineConfig(
    vad_threshold=0.6,           # stricter VAD
    min_segment_duration=0.5,    # minimum segment length
    collar=0.2,                  # merge adjacent segments within 0.2s
    max_speakers=5,              # upper bound on speaker count
)

pipeline = DiarizationPipeline(config=config)
result = pipeline.process("audio.wav")
```

### Export to RTTM Format

```python
result = pipeline.process("audio.wav")
rttm_str = result.to_rttm()  # Standard RTTM format for evaluation
print(rttm_str)
```

## Components

### Voice Activity Detection (VAD)
Detects speech regions in audio using energy-based analysis with optional thresholding. Removes silence and non-speech segments before downstream processing.

### Speaker Embeddings
Extracts fixed-dimensional speaker embeddings from each speech region. These embeddings are trained to be speaker-discriminative and robust to channel variation.

### Spectral Clustering
Clusters speaker embeddings using spectral clustering with automatic speaker count estimation via eigengap analysis. Adapts to variable speaker counts without manual specification.

### Segment Refinement
Post-processing to enforce temporal constraints and merge fragmented speaker turns caused by brief pauses. Supports collar-based merging and speaker label assignment.

## Testing

Run the comprehensive test suite (90+ tests):

```bash
pytest tests/ -v
```

Test coverage includes:
- Unit tests for all classes and methods
- Edge case handling (empty inputs, boundary conditions)
- Integration tests for end-to-end pipeline
- Determinism and reproducibility checks

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on development setup, testing, and submitting changes.

## Evaluation Notes

The repo exposes diarization outputs in JSON and RTTM formats so you can evaluate against your own labeled audio. Metrics depend heavily on the dataset, channel quality, language, number of speakers, and overlap rate.

Use DER, speaker attribution accuracy, and boundary tolerance as the core evaluation lens.

## API Reference

### DiarizationPipeline

Main entry point for diarization.

```python
pipeline = DiarizationPipeline(
    config: Optional[PipelineConfig] = None,
    model_dir: Optional[Path] = None,
)

result = pipeline.process(audio_path: str | Path) -> DiarizationResult
results = pipeline.process_batch(audio_paths: list[str | Path]) -> list[DiarizationResult]
pipeline.initialize() -> None
```

### DiarizationResult

Output structure containing speaker-labeled segments.

```python
result.segments: list[Segment]              # speaker-labeled segments
result.num_speakers: int                    # detected speaker count
result.total_duration: float                # total audio duration
result.speakers() -> list[str]              # unique speaker labels
result.segments_for(speaker: str) -> list[Segment]  # segments by speaker
result.to_rttm() -> str                     # export as RTTM format
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

Built by [Anmol Dhingra](https://github.com/anmoldhingra1), founder of [Rerato](https://trivana.ai).
