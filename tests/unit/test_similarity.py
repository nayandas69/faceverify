"""Tests for similarity metrics."""

import pytest
import numpy as np
from faceverify.similarity.metrics import (
    cosine_similarity,
    euclidean_distance,
    manhattan_distance,
)
from faceverify.similarity.engine import SimilarityEngine


class TestCosineSmiliarity:
    """Tests for cosine similarity metric."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity of 1."""
        vec = np.array([1.0, 2.0, 3.0])
        sim, dist = cosine_similarity(vec, vec)

        assert sim == pytest.approx(1.0, abs=0.001)
        assert dist == pytest.approx(0.0, abs=0.001)

    def test_opposite_vectors(self):
        """Opposite vectors should have low similarity."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])

        sim, dist = cosine_similarity(vec1, vec2)

        assert sim == pytest.approx(0.0, abs=0.001)
        assert dist == pytest.approx(2.0, abs=0.001)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity of 0.5."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        sim, dist = cosine_similarity(vec1, vec2)

        assert sim == pytest.approx(0.5, abs=0.001)

    def test_normalized_embeddings(self, sample_embedding, sample_embedding_similar):
        """Test with normalized embeddings."""
        sim, dist = cosine_similarity(sample_embedding, sample_embedding_similar)

        assert 0 <= sim <= 1
        assert dist >= 0


class TestEuclideanDistance:
    """Tests for Euclidean distance metric."""

    def test_identical_vectors(self):
        """Identical vectors should have distance of 0."""
        vec = np.array([1.0, 2.0, 3.0])
        sim, dist = euclidean_distance(vec, vec)

        assert dist == pytest.approx(0.0, abs=0.001)
        assert sim == pytest.approx(1.0, abs=0.001)

    def test_known_distance(self):
        """Test with known distance."""
        vec1 = np.array([0.0, 0.0])
        vec2 = np.array([3.0, 4.0])

        sim, dist = euclidean_distance(vec1, vec2)

        assert dist == pytest.approx(5.0, abs=0.001)


class TestSimilarityEngine:
    """Tests for SimilarityEngine."""

    def test_create_cosine_engine(self):
        """Test creating engine with cosine metric."""
        engine = SimilarityEngine(metric="cosine")
        assert engine.metric == "cosine"

    def test_create_euclidean_engine(self):
        """Test creating engine with euclidean metric."""
        engine = SimilarityEngine(metric="euclidean")
        assert engine.metric == "euclidean"

    def test_invalid_metric(self):
        """Test that invalid metric raises error."""
        with pytest.raises(ValueError):
            SimilarityEngine(metric="invalid")

    def test_compute(self, sample_embedding, sample_embedding_similar):
        """Test compute method."""
        engine = SimilarityEngine(metric="cosine")
        sim, dist = engine.compute(sample_embedding, sample_embedding_similar)

        assert isinstance(sim, float)
        assert isinstance(dist, float)
        assert 0 <= sim <= 1
