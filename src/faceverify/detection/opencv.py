"""OpenCV-based face detector implementation."""

from typing import List
import numpy as np
import cv2

from faceverify.detection.base import BaseDetector
from faceverify.core.result import DetectionResult, BoundingBox


class OpenCVDetector(BaseDetector):
    """
    OpenCV Haar Cascade face detector.

    This is a lightweight, fast detector suitable for
    real-time applications with limited resources.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple = (30, 30),
        **kwargs,
    ):
        """
        Initialize OpenCV detector.

        Args:
            confidence_threshold: Minimum confidence (based on size)
            scale_factor: Image scale factor for detection
            min_neighbors: Minimum neighbors for detection
            min_size: Minimum face size
        """
        super().__init__(confidence_threshold=confidence_threshold)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    @property
    def name(self) -> str:
        return "opencv"

    def _load_model(self) -> None:
        """Load OpenCV Haar Cascade model."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._model = cv2.CascadeClassifier(cascade_path)

        if self._model.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")

    def _detect_impl(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect faces using OpenCV Haar Cascade.

        Args:
            image: RGB image as numpy array

        Returns:
            List of DetectionResult objects
        """
        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = self._model.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        results = []
        image_area = image.shape[0] * image.shape[1]

        for x, y, w, h in faces:
            bbox = BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h))

            # Estimate confidence based on face size relative to image
            face_area = w * h
            size_ratio = face_area / image_area
            confidence = min(1.0, size_ratio * 10 + 0.5)  # Heuristic

            # Crop face region
            face_image = image[y : y + h, x : x + w].copy()

            results.append(
                DetectionResult(
                    bounding_box=bbox,
                    confidence=confidence,
                    landmarks=None,  # Haar Cascade doesn't provide landmarks
                    face_image=face_image,
                )
            )

        return results
