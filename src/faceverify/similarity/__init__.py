"""Similarity computation module."""

from faceverify.similarity.engine import SimilarityEngine
from faceverify.similarity.metrics import (
    cosine_similarity,
    euclidean_distance,
    manhattan_distance,
)

__all__ = [
    "SimilarityEngine",
    "cosine_similarity",
    "euclidean_distance",
    "manhattan_distance",
]
