"""Result dataclasses for face verification operations."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import numpy as np


@dataclass
class BoundingBox:
    """Represents a face bounding box."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x1(self) -> int:
        return self.x

    @property
    def y1(self) -> int:
        return self.y

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Landmarks:
    """Facial landmarks."""

    left_eye: Optional[Tuple[float, float]] = None
    right_eye: Optional[Tuple[float, float]] = None
    nose: Optional[Tuple[float, float]] = None
    left_mouth: Optional[Tuple[float, float]] = None
    right_mouth: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Optional[Tuple[float, float]]]:
        return {
            "left_eye": self.left_eye,
            "right_eye": self.right_eye,
            "nose": self.nose,
            "left_mouth": self.left_mouth,
            "right_mouth": self.right_mouth,
        }


@dataclass
class DetectionResult:
    """Result of face detection."""

    bounding_box: BoundingBox
    confidence: float
    landmarks: Optional[Landmarks] = None
    face_image: Optional[np.ndarray] = None
    aligned_face: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )


@dataclass
class EmbeddingResult:
    """Result of face embedding extraction."""

    embedding: np.ndarray
    model_name: str
    dimension: int
    normalized: bool
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {self.embedding.shape[0]}"
            )


@dataclass
class VerificationResult:
    """Result of face verification between two images."""

    verified: bool
    confidence: float
    similarity: float
    distance: float
    threshold: float
    detector_backend: str
    embedding_model: str
    similarity_metric: str
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )
        if not 0 <= self.similarity <= 1:
            raise ValueError(
                f"Similarity must be between 0 and 1, got {self.similarity}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "verified": self.verified,
            "confidence": self.confidence,
            "similarity": self.similarity,
            "distance": self.distance,
            "threshold": self.threshold,
            "detector_backend": self.detector_backend,
            "embedding_model": self.embedding_model,
            "similarity_metric": self.similarity_metric,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        status = "✓ VERIFIED" if self.verified else "✗ NOT VERIFIED"
        return (
            f"{status}\n"
            f"  Confidence: {self.confidence:.2%}\n"
            f"  Similarity: {self.similarity:.4f}\n"
            f"  Threshold: {self.threshold:.4f}\n"
            f"  Time: {self.processing_time:.3f}s"
        )


@dataclass
class IdentificationResult:
    """Result of face identification (1:N matching)."""

    query_image: str
    matches: List[Dict[str, Any]]
    best_match: Optional[Dict[str, Any]]
    total_candidates: int
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.matches:
            self.best_match = max(self.matches, key=lambda x: x.get("similarity", 0))
        else:
            self.best_match = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_image": self.query_image,
            "matches": self.matches,
            "best_match": self.best_match,
            "total_candidates": self.total_candidates,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat(),
        }
