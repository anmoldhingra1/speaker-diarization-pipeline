"""Spectral clustering for speaker embeddings."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SpectralClusterer:
    """Cluster speaker embeddings using spectral clustering.

    Automatically estimates the number of speakers when not specified,
    using eigengap analysis on the affinity matrix.

    Args:
        max_speakers: Upper bound on number of speakers.
        min_speakers: Lower bound on number of speakers.
        threshold: Affinity threshold for building the similarity graph.
    """

    def __init__(
        self,
        max_speakers: int = 10,
        min_speakers: int = 2,
        threshold: float = 0.5,
    ) -> None:
        self.max_speakers = max_speakers
        self.min_speakers = min_speakers
        self.threshold = threshold

    def cluster(self, embeddings: np.ndarray) -> list[int]:
        """Assign speaker labels to embeddings.

        Args:
            embeddings: Array of shape (n_segments, embedding_dim).

        Returns:
            List of integer speaker labels, one per segment.
        """
        n = len(embeddings)
        if n == 0:
            return []
        if n == 1:
            return [0]

        # Build affinity matrix (cosine similarity)
        affinity = self._build_affinity(embeddings)

        # Estimate number of speakers via eigengap
        n_speakers = self._estimate_speakers(affinity)
        logger.info("Estimated %d speakers from %d segments", n_speakers, n)

        # Spectral clustering
        labels = self._spectral_cluster(affinity, n_speakers)
        return labels

    def _build_affinity(self, embeddings: np.ndarray) -> np.ndarray:
        """Build cosine similarity affinity matrix."""
        similarity = embeddings @ embeddings.T
        np.fill_diagonal(similarity, 1.0)
        # Apply threshold
        affinity = np.where(similarity > self.threshold, similarity, 0.0)
        return affinity

    def _estimate_speakers(self, affinity: np.ndarray) -> int:
        """Estimate speaker count using eigengap heuristic."""
        n = len(affinity)
        if n <= self.min_speakers:
            return self.min_speakers

        # Compute Laplacian eigenvalues
        degree = np.diag(affinity.sum(axis=1))
        laplacian = degree - affinity
        eigenvalues = np.sort(np.real(np.linalg.eigvalsh(laplacian)))

        # Find largest eigengap
        max_k = min(self.max_speakers, n)
        gaps = np.diff(eigenvalues[1:max_k + 1])

        if len(gaps) == 0:
            return self.min_speakers

        k = int(np.argmax(gaps)) + 2  # +2 because we started from index 1
        return max(self.min_speakers, min(k, self.max_speakers))

    def _spectral_cluster(
        self, affinity: np.ndarray, n_clusters: int
    ) -> list[int]:
        """Run spectral clustering with given number of clusters."""
        n = len(affinity)

        # Normalized Laplacian
        degree = affinity.sum(axis=1)
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
        d_mat = np.diag(d_inv_sqrt)
        laplacian_norm = np.eye(n) - d_mat @ affinity @ d_mat

        # Get bottom-k eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian_norm)
        features = eigenvectors[:, :n_clusters]

        # Normalize rows
        row_norms = np.linalg.norm(features, axis=1, keepdims=True)
        row_norms = np.maximum(row_norms, 1e-8)
        features = features / row_norms

        # K-means on spectral features
        labels = self._kmeans(features, n_clusters)
        return labels

    def _kmeans(
        self, data: np.ndarray, k: int, max_iter: int = 100
    ) -> list[int]:
        """Simple k-means clustering."""
        n = len(data)
        rng = np.random.RandomState(42)

        # Initialize centroids with k-means++
        centroids = [data[rng.randint(n)]]
        for _ in range(1, k):
            dists = np.min(
                [np.sum((data - c) ** 2, axis=1) for c in centroids], axis=0
            )
            probs = dists / (dists.sum() + 1e-8)
            centroids.append(data[rng.choice(n, p=probs)])
        centroids = np.array(centroids)

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            # Assign
            dists = np.array(
                [np.sum((data - c) ** 2, axis=1) for c in centroids]
            )
            new_labels = np.argmin(dists, axis=0)

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            # Update centroids
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centroids[j] = data[mask].mean(axis=0)

        return labels.tolist()
