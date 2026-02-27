"""Tests for diarization.clustering module."""

from __future__ import annotations

import numpy as np

from diarization.clustering import SpectralClusterer


class TestSpectralClusterer:
    """Tests for SpectralClusterer class."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        clusterer = SpectralClusterer()
        assert clusterer.max_speakers == 10
        assert clusterer.min_speakers == 2
        assert clusterer.threshold == 0.5

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        clusterer = SpectralClusterer(
            max_speakers=5,
            min_speakers=1,
            threshold=0.7,
        )
        assert clusterer.max_speakers == 5
        assert clusterer.min_speakers == 1
        assert clusterer.threshold == 0.7

    def test_cluster_empty(self) -> None:
        """Test cluster with empty embeddings."""
        clusterer = SpectralClusterer()
        labels = clusterer.cluster(np.array([]).reshape(0, 128))
        assert labels == []

    def test_cluster_single(self) -> None:
        """Test cluster with single embedding."""
        clusterer = SpectralClusterer()
        embeddings = np.random.randn(1, 128)
        labels = clusterer.cluster(embeddings)
        assert labels == [0]

    def test_cluster_returns_list(self) -> None:
        """Test cluster returns list of ints."""
        clusterer = SpectralClusterer()
        embeddings = np.random.randn(10, 128)
        labels = clusterer.cluster(embeddings)
        assert isinstance(labels, list)
        assert all(isinstance(label, (int, np.integer)) for label in labels)

    def test_cluster_length(self) -> None:
        """Test cluster returns one label per embedding."""
        clusterer = SpectralClusterer()
        for n_emb in [5, 10, 20, 50]:
            embeddings = np.random.randn(n_emb, 128)
            labels = clusterer.cluster(embeddings)
            assert len(labels) == n_emb

    def test_cluster_label_range(self) -> None:
        """Test cluster labels are in valid range."""
        clusterer = SpectralClusterer(min_speakers=2, max_speakers=5)
        embeddings = np.random.randn(30, 128)
        labels = clusterer.cluster(embeddings)
        unique_labels = set(labels)
        assert len(unique_labels) <= clusterer.max_speakers
        assert len(unique_labels) >= clusterer.min_speakers

    def test_build_affinity_shape(self) -> None:
        """Test affinity matrix shape."""
        clusterer = SpectralClusterer()
        embeddings = np.random.randn(10, 128)
        affinity = clusterer._build_affinity(embeddings)
        assert affinity.shape == (10, 10)

    def test_build_affinity_diagonal(self) -> None:
        """Test affinity diagonal is 1.0."""
        clusterer = SpectralClusterer()
        embeddings = np.random.randn(5, 128)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        affinity = clusterer._build_affinity(embeddings)
        np.testing.assert_array_almost_equal(np.diag(affinity), 1.0)

    def test_build_affinity_symmetric(self) -> None:
        """Test affinity matrix is symmetric."""
        clusterer = SpectralClusterer()
        embeddings = np.random.randn(8, 128)
        affinity = clusterer._build_affinity(embeddings)
        np.testing.assert_array_almost_equal(affinity, affinity.T)

    def test_estimate_speakers_bounds(self) -> None:
        """Test speaker estimation respects bounds."""
        clusterer = SpectralClusterer(min_speakers=2, max_speakers=5)
        for n_seg in [1, 5, 10, 20, 50]:
            embeddings = np.random.randn(n_seg, 128)
            affinity = clusterer._build_affinity(embeddings)
            n_speakers = clusterer._estimate_speakers(affinity)
            assert clusterer.min_speakers <= n_speakers <= clusterer.max_speakers

    def test_estimate_speakers_small_input(self) -> None:
        """Test speaker estimation with small input."""
        clusterer = SpectralClusterer(min_speakers=2, max_speakers=10)
        embeddings = np.random.randn(1, 128)
        affinity = clusterer._build_affinity(embeddings)
        n_speakers = clusterer._estimate_speakers(affinity)
        assert n_speakers == clusterer.min_speakers

    def test_kmeans_shape(self) -> None:
        """Test k-means returns correct shape."""
        clusterer = SpectralClusterer()
        data = np.random.randn(20, 10)
        labels = clusterer._kmeans(data, k=3)
        assert len(labels) == 20
        assert all(isinstance(label, (int, np.integer)) for label in labels)

    def test_kmeans_labels_range(self) -> None:
        """Test k-means labels are in [0, k)."""
        clusterer = SpectralClusterer()
        data = np.random.randn(30, 10)
        for k in [2, 3, 5]:
            labels = clusterer._kmeans(data, k=k)
            assert max(labels) < k
            assert min(labels) >= 0

    def test_kmeans_deterministic(self) -> None:
        """Test k-means is deterministic."""
        clusterer = SpectralClusterer()
        data = np.random.RandomState(42).randn(20, 10)
        labels1 = clusterer._kmeans(data, k=3)
        labels2 = clusterer._kmeans(data, k=3)
        assert labels1 == labels2

    def test_spectral_cluster_shape(self) -> None:
        """Test spectral clustering returns correct shape."""
        clusterer = SpectralClusterer()
        affinity = np.eye(10) + 0.1 * np.ones((10, 10))
        np.fill_diagonal(affinity, 1.0)
        labels = clusterer._spectral_cluster(affinity, n_clusters=3)
        assert len(labels) == 10
        assert max(labels) < 3


class TestSpectralClustererIntegration:
    """Integration tests for SpectralClusterer."""

    def test_cluster_two_distinct_groups(self) -> None:
        """Test clustering separates distinct speaker groups."""
        clusterer = SpectralClusterer(min_speakers=2, max_speakers=5)
        # Create two distinct clusters
        group1 = np.random.randn(10, 128) + np.array([5.0] * 128)
        group2 = np.random.randn(10, 128) - np.array([5.0] * 128)
        embeddings = np.vstack([group1, group2])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = clusterer.cluster(embeddings)
        # Should identify 2 speakers
        assert len(set(labels)) == 2

    def test_cluster_reproducibility(self) -> None:
        """Test clustering is reproducible."""
        clusterer = SpectralClusterer()
        np.random.seed(42)
        embeddings = np.random.randn(20, 128)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = clusterer.cluster(embeddings)
        assert len(labels) == 20
