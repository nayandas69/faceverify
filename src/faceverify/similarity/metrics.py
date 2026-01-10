"""Similarity and distance metrics for face embeddings."""

import numpy as np
from typing import Tuple


def cosine_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute cosine similarity between two embeddings.

    Cosine similarity measures the cosine of the angle between
    two vectors, ranging from -1 to 1 (1 = identical).

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Tuple of (similarity, distance) where:
            - similarity: 0 to 1 (higher = more similar)
            - distance: 0 to 2 (lower = more similar)
    """
    # Ensure vectors are normalized
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 > 0:
        embedding1 = embedding1 / norm1
    if norm2 > 0:
        embedding2 = embedding2 / norm2

    # Cosine similarity
    dot_product = np.dot(embedding1, embedding2)
    similarity = float((dot_product + 1) / 2)  # Map from [-1,1] to [0,1]

    # Cosine distance
    distance = float(1 - dot_product)

    return similarity, distance


def euclidean_distance(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Euclidean distance between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Tuple of (similarity, distance) where:
            - similarity: 0 to 1 (higher = more similar)
            - distance: >= 0 (lower = more similar)
    """
    # L2 distance
    distance = float(np.linalg.norm(embedding1 - embedding2))

    # Convert to similarity (0 to 1)
    # Using exponential decay: sim = exp(-distance/scale)
    similarity = float(np.exp(-distance / 2.0))

    return similarity, distance


def manhattan_distance(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Manhattan (L1) distance between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Tuple of (similarity, distance)
    """
    # L1 distance
    distance = float(np.sum(np.abs(embedding1 - embedding2)))

    # Convert to similarity
    similarity = float(np.exp(-distance / 10.0))

    return similarity, distance
