"""Tests for diarization.pipeline module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from diarization.pipeline import DiarizationPipeline
from diarization.types import PipelineConfig, DiarizationResult, Segment


class TestDiarizationPipeline:
    """Tests for DiarizationPipeline class."""

    def test_init_defaults(self) -> None:
        """Test default pipeline initialization."""
        pipeline = DiarizationPipeline()
        assert pipeline.config is not None
        assert pipeline.model_dir == Path("models")
        assert pipeline._is_initialized is False

    def test_init_custom_config(self) -> None:
        """Test pipeline with custom config."""
        config = PipelineConfig(vad_threshold=0.6, max_speakers=5)
        pipeline = DiarizationPipeline(config=config)
        assert pipeline.config.vad_threshold == 0.6
        assert pipeline.config.max_speakers == 5

    def test_init_custom_model_dir(self) -> None:
        """Test pipeline with custom model directory."""
        model_dir = Path("/custom/models")
        pipeline = DiarizationPipeline(model_dir=model_dir)
        assert pipeline.model_dir == model_dir

    def test_initialize_idempotent(self) -> None:
        """Test initialize is idempotent."""
        pipeline = DiarizationPipeline()
        pipeline.initialize()
        assert pipeline._is_initialized is True
        pipeline.initialize()
        assert pipeline._is_initialized is True

    def test_process_file_not_found(self) -> None:
        """Test process raises FileNotFoundError for missing file."""
        pipeline = DiarizationPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.process("nonexistent_file.wav")

    def test_process_initializes_pipeline(self) -> None:
        """Test process initializes pipeline."""
        pipeline = DiarizationPipeline()
        assert pipeline._is_initialized is False
        # Mock VAD to return empty
        with patch.object(pipeline._vad, 'detect', return_value=[]):
            result = pipeline.process(__file__)
            # Should have initialized
            assert pipeline._is_initialized is True

    def test_process_returns_result(self) -> None:
        """Test process returns DiarizationResult."""
        pipeline = DiarizationPipeline()
        with patch.object(pipeline._vad, 'detect', return_value=[]):
            result = pipeline.process(__file__)
            assert isinstance(result, DiarizationResult)

    def test_process_empty_vad(self) -> None:
        """Test process with no speech regions."""
        pipeline = DiarizationPipeline()
        with patch.object(pipeline._vad, 'detect', return_value=[]):
            result = pipeline.process(__file__)
            assert result.segments == []
            assert result.num_speakers == 0

    def test_process_batch_empty(self) -> None:
        """Test process_batch with empty list."""
        pipeline = DiarizationPipeline()
        results = pipeline.process_batch([])
        assert results == []

    def test_process_batch_returns_list(self) -> None:
        """Test process_batch returns list of results."""
        pipeline = DiarizationPipeline()
        with patch.object(pipeline._vad, 'detect', return_value=[]):
            results = pipeline.process_batch([__file__, __file__])
            assert isinstance(results, list)
            assert len(results) == 2
            assert all(isinstance(r, DiarizationResult) for r in results)


class TestDiarizationPipelineIntegration:
    """Integration tests for DiarizationPipeline."""

    def test_end_to_end_pipeline(self) -> None:
        """Test end-to-end pipeline execution."""
        from diarization.types import SpeechRegion

        pipeline = DiarizationPipeline()

        # Mock components
        regions = [
            SpeechRegion(start=0.0, end=1.0, confidence=0.9),
            SpeechRegion(start=1.0, end=2.0, confidence=0.9),
        ]

        with patch.object(pipeline._vad, 'detect', return_value=regions):
            with patch.object(
                pipeline._embedder,
                'extract',
                return_value=__import__('numpy').eye(2, 128),
            ):
                with patch.object(
                    pipeline._clusterer, 'cluster', return_value=[0, 1]
                ):
                    result = pipeline.process(__file__)
                    assert result.num_speakers == 2
                    assert len(result.segments) == 2


class TestDiarizationPipelineEdgeCases:
    """Edge case tests for DiarizationPipeline."""

    def test_process_with_path_string(self) -> None:
        """Test process accepts string path."""
        pipeline = DiarizationPipeline()
        with patch.object(pipeline._vad, 'detect', return_value=[]):
            result = pipeline.process(__file__)
            assert isinstance(result, DiarizationResult)

    def test_process_with_path_object(self) -> None:
        """Test process accepts Path object."""
        pipeline = DiarizationPipeline()
        with patch.object(pipeline._vad, 'detect', return_value=[]):
            result = pipeline.process(Path(__file__))
            assert isinstance(result, DiarizationResult)

    def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialization flag."""
        pipeline = DiarizationPipeline()
        assert not pipeline._is_initialized
        pipeline.initialize()
        assert pipeline._is_initialized
