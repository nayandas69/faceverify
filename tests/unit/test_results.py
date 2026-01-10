"""Tests for result dataclasses."""

import pytest
import numpy as np
from faceverify.core.result import (
    BoundingBox,
    Landmarks,
    DetectionResult,
    EmbeddingResult,
    VerificationResult,
)


class TestBoundingBox:
    """Tests for BoundingBox."""

    def test_properties(self):
        """Test bounding box properties."""
        bbox = BoundingBox(x=10, y=20, width=100, height=150)

        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 110
        assert bbox.y2 == 170
        assert bbox.center == (60, 95)
        assert bbox.area == 15000

    def test_to_tuple(self):
        """Test conversion to tuple."""
        bbox = BoundingBox(x=10, y=20, width=100, height=150)
        assert bbox.to_tuple() == (10, 20, 100, 150)

    def test_to_xyxy(self):
        """Test conversion to xyxy format."""
        bbox = BoundingBox(x=10, y=20, width=100, height=150)
        assert bbox.to_xyxy() == (10, 20, 110, 170)


class TestDetectionResult:
    """Tests for DetectionResult."""

    def test_valid_creation(self):
        """Test creating valid detection result."""
        bbox = BoundingBox(x=10, y=20, width=100, height=100)
        result = DetectionResult(bounding_box=bbox, confidence=0.95)

        assert result.confidence == 0.95
        assert result.bounding_box == bbox

    def test_invalid_confidence(self):
        """Test that invalid confidence raises error."""
        bbox = BoundingBox(x=10, y=20, width=100, height=100)

        with pytest.raises(ValueError):
            DetectionResult(bounding_box=bbox, confidence=1.5)


class TestVerificationResult:
    """Tests for VerificationResult."""

    def test_valid_creation(self):
        """Test creating valid verification result."""
        result = VerificationResult(
            verified=True,
            confidence=0.85,
            similarity=0.78,
            distance=0.22,
            threshold=0.65,
            detector_backend="mtcnn",
            embedding_model="facenet",
            similarity_metric="cosine",
            processing_time=0.5,
        )

        assert result.verified is True
        assert result.confidence == 0.85

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = VerificationResult(
            verified=True,
            confidence=0.85,
            similarity=0.78,
            distance=0.22,
            threshold=0.65,
            detector_backend="mtcnn",
            embedding_model="facenet",
            similarity_metric="cosine",
            processing_time=0.5,
        )

        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["verified"] is True
        assert d["confidence"] == 0.85

    def test_string_representation(self):
        """Test string representation."""
        result = VerificationResult(
            verified=True,
            confidence=0.85,
            similarity=0.78,
            distance=0.22,
            threshold=0.65,
            detector_backend="mtcnn",
            embedding_model="facenet",
            similarity_metric="cosine",
            processing_time=0.5,
        )

        s = str(result)
        assert "VERIFIED" in s
        assert "Confidence" in s
