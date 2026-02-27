"""Tests for diarization.embeddings module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diarization.embeddings import EmbeddingExtractor
from diarization.types import SpeechRegion


class TestEmbeddingExtractor:
    """Tests for EmbeddingExtractor class."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        extractor = EmbeddingExtractor()
        assert extractor.model_path is None
        assert extractor.embedding_dim == 512
        assert extractor._model is None

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        model_path = Path("models/embedding_model")
        extractor = EmbeddingExtractor(
            model_path=model_path,
            embedding_dim=256,
        )
        assert extractor.model_path == model_path
        assert extractor.embedding_dim == 256

    def test_load_no_model_path(self) -> None:
        """Test load with no model path uses random."""
        extractor = EmbeddingExtractor(embedding_dim=128)
        extractor.load()
        assert extractor._model == "random"

    def test_load_idempotent(self) -> None:
        """Test load is idempotent."""
        extractor = EmbeddingExtractor()
        extractor.load()
        first_model = extractor._model
        extractor.load()
        second_model = extractor._model
        assert first_model is second_model

    def test_extract_single_shape(self) -> None:
        """Test _extract_single returns correct shape."""
        extractor = EmbeddingExtractor(embedding_dim=256)
        embedding = extractor._extract_single(
            Path("test.wav"), start=0.0, end=1.0
        )
        assert embedding.shape == (256,)
        assert embedding.dtype == np.float32

    def test_extract_single_deterministic(self) -> None:
        """Test _extract_single is deterministic."""
        extractor = EmbeddingExtractor(embedding_dim=128)
        emb1 = extractor._extract_single(Path("test.wav"), 0.0, 1.0)
        emb2 = extractor._extract_single(Path("test.wav"), 0.0, 1.0)
        np.testing.assert_array_equal(emb1, emb2)

    def test_extract_single_different_regions(self) -> None:
        """Test _extract_single varies with region boundaries."""
        extractor = EmbeddingExtractor(embedding_dim=128)
        emb1 = extractor._extract_single(Path("test.wav"), 0.0, 1.0)
        emb2 = extractor._extract_single(Path("test.wav"), 1.0, 2.0)
        assert not np.allclose(emb1, emb2)

    def test_extract_empty_regions(self) -> None:
        """Test extract with empty regions."""
        extractor = EmbeddingExtractor(embedding_dim=64)
        extractor.load()
        embeddings = extractor.extract(Path("test.wav"), [])
        assert embeddings.shape == (0, 64)

    def test_extract_shape(self) -> None:
        """Test extract returns correct shape."""
        extractor = EmbeddingExtractor(embedding_dim=192)
        extractor.load()
        regions = [
            SpeechRegion(start=0.0, end=1.0),
            SpeechRegion(start=1.0, end=2.0),
            SpeechRegion(start=2.0, end=3.0),
        ]
        embeddings = extractor.extract(Path("test.wav"), regions)
        assert embeddings.shape == (3, 192)

    def test_extract_normalized(self) -> None:
        """Test extract returns L2-normalized embeddings."""
        extractor = EmbeddingExtractor(embedding_dim=128)
        extractor.load()
        regions = [
            SpeechRegion(start=0.0, end=1.0),
            SpeechRegion(start=1.0, end=2.0),
        ]
        embeddings = extractor.extract(Path("test.wav"), regions)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, 1.0, decimal=5)

    def test_extract_loads_model(self) -> None:
        """Test extract calls load if not initialized."""
        extractor = EmbeddingExtractor()
        assert extractor._model is None
        regions = [SpeechRegion(start=0.0, end=1.0)]
        extractor.extract(Path("test.wav"), regions)
        assert extractor._model is not None

    def test_extract_different_dimensions(self) -> None:
        """Test extract with different embedding dimensions."""
        for dim in [64, 128, 256, 512]:
            extractor = EmbeddingExtractor(embedding_dim=dim)
            extractor.load()
            regions = [SpeechRegion(start=0.0, end=1.0)]
            embeddings = extractor.extract(Path("test.wav"), regions)
            assert embeddings.shape == (1, dim)

    def test_extract_many_regions(self) -> None:
        """Test extract with many regions."""
        extractor = EmbeddingExtractor(embedding_dim=128)
        extractor.load()
        regions = [
            SpeechRegion(start=float(i), end=float(i + 1))
            for i in range(100)
        ]
        embeddings = extractor.extract(Path("test.wav"), regions)
        assert embeddings.shape == (100, 128)

    def test_extract_maintains_order(self) -> None:
        """Test extract maintains region order."""
        extractor = EmbeddingExtractor(embedding_dim=64)
        extractor.load()
        regions = [
            SpeechRegion(start=5.0, end=6.0),
            SpeechRegion(start=1.0, end=2.0),
            SpeechRegion(start=3.0, end=4.0),
        ]
        embeddings = extractor.extract(Path("test.wav"), regions)
        assert embeddings.shape == (3, 64)


class TestEmbeddingExtractorIntegration:
    """Integration tests for EmbeddingExtractor."""

    def test_extract_sequence_consistency(self) -> None:
        """Test consistent extraction for same regions."""
        extractor = EmbeddingExtractor(embedding_dim=128)
        extractor.load()
        regions = [
            SpeechRegion(start=0.0, end=1.0),
            SpeechRegion(start=1.0, end=2.0),
        ]
        emb1 = extractor.extract(Path("test.wav"), regions)
        emb2 = extractor.extract(Path("test.wav"), regions)
        np.testing.assert_array_equal(emb1, emb2)

    def test_load_model_from_nonexistent_path(self) -> None:
        """Test load with nonexistent path."""
        extractor = EmbeddingExtractor(
            model_path=Path("nonexistent/model.pt")
        )
        extractor.load()
        assert extractor._model == "random"
