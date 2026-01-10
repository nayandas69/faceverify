"""Similarity engine for comparing face embeddings."""

from typing import Tuple, Callable
import numpy as np

from faceverify.similarity.metrics import (
    cosine_similarity,
    euclidean_distance,
    manhattan_distance,
)


class SimilarityEngine:
    """
    Engine for computing similarity between face embeddings.

    Supports multiple distance/similarity metrics and provides
    a consistent interface for comparison operations.
    """

    METRICS = {
        "cosine": cosine_similarity,
        "euclidean": euclidean_distance,
        "manhattan": manhattan_distance,
    }

    def __init__(self, metric: str = "cosine"):
        """
        Initialize similarity engine.

        Args:
            metric: Similarity metric to use
        """
        if metric not in self.METRICS:
            raise ValueError(
                f"Unknown metric: {metric}. " f"Supported: {list(self.METRICS.keys())}"
            )

        self.metric = metric
        self._compute_fn: Callable = self.METRICS[metric]

    def compute(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding

        Returns:
            Tuple of (similarity, distance)
        """
        return self._compute_fn(embedding1, embedding2)

    def compute_matrix(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pairwise similarities between two sets of embeddings.

        Args:
            embeddings1: Array of shape (n, dim)
            embeddings2: Array of shape (m, dim)

        Returns:
            Tuple of (similarity_matrix, distance_matrix)
                both of shape (n, m)
        """
        n = embeddings1.shape[0]
        m = embeddings2.shape[0]

        similarities = np.zeros((n, m))
        distances = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                sim, dist = self.compute(embeddings1[i], embeddings2[j])
                similarities[i, j] = sim
                distances[i, j] = dist

        return similarities, distances
