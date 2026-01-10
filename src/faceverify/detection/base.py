"""Abstract base class for face detectors."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from faceverify.core.result import DetectionResult


class BaseDetector(ABC):
    """
    Abstract base class for face detection backends.

    All face detectors must implement this interface to ensure
    consistent behavior across different detection methods.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        **kwargs,
    ):
        """
        Initialize detector.

        Args:
            confidence_threshold: Minimum confidence for detections
            **kwargs: Backend-specific parameters
        """
        self.confidence_threshold = confidence_threshold
        self._model = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the detector."""
        pass

    @abstractmethod
    def _load_model(self) -> None:
        """Load the detection model."""
        pass

    @abstractmethod
    def _detect_impl(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Implementation of face detection.

        Args:
            image: Input image as numpy array (RGB format)

        Returns:
            List of DetectionResult objects
        """
        pass

    def detect(
        self,
        image: np.ndarray,
        return_scores: bool = False,
    ) -> List[DetectionResult]:
        """
        Detect faces in an image.

        Args:
            image: Input image as numpy array (RGB format)
            return_scores: Whether to include confidence scores

        Returns:
            List of DetectionResult objects for each detected face
        """
        if self._model is None:
            self._load_model()

        detections = self._detect_impl(image)

        # Filter by confidence threshold
        detections = [
            d for d in detections if d.confidence >= self.confidence_threshold
        ]

        return detections

    def detect_largest(self, image: np.ndarray) -> Optional[DetectionResult]:
        """
        Detect the largest face in an image.

        Args:
            image: Input image

        Returns:
            DetectionResult for the largest face, or None if no face found
        """
        detections = self.detect(image)

        if not detections:
            return None

        return max(detections, key=lambda d: d.bounding_box.area)
