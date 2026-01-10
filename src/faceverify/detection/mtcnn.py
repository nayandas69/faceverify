"""MTCNN face detector implementation."""

from typing import List, Optional, Tuple
import numpy as np

from faceverify.detection.base import BaseDetector
from faceverify.core.result import DetectionResult, BoundingBox, Landmarks


class MTCNNDetector(BaseDetector):
    """
    Multi-task Cascaded Convolutional Networks (MTCNN) face detector.

    MTCNN is a popular face detection model that provides both
    bounding boxes and facial landmarks.

    Reference:
        Zhang et al., "Joint Face Detection and Alignment using
        Multi-task Cascaded Convolutional Networks", 2016
    """

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        min_face_size: int = 20,
        scale_factor: float = 0.709,
        **kwargs,
    ):
        """
        Initialize MTCNN detector.

        Args:
            confidence_threshold: Minimum confidence for detections
            min_face_size: Minimum face size to detect
            scale_factor: Scale factor for image pyramid
        """
        super().__init__(confidence_threshold=confidence_threshold)
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor

    @property
    def name(self) -> str:
        return "mtcnn"

    def _load_model(self) -> None:
        """Load MTCNN model."""
        try:
            from mtcnn import MTCNN

            self._model = MTCNN(
                min_face_size=self.min_face_size,
                scale_factor=self.scale_factor,
            )
        except ImportError:
            raise ImportError("MTCNN not installed. Install with: pip install mtcnn")

    def _detect_impl(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect faces using MTCNN.

        Args:
            image: RGB image as numpy array

        Returns:
            List of DetectionResult objects
        """
        results = []

        # MTCNN detect returns list of dicts
        detections = self._model.detect_faces(image)

        for detection in detections:
            confidence = detection["confidence"]

            # Extract bounding box
            x, y, w, h = detection["box"]

            # Ensure non-negative coordinates
            x = max(0, x)
            y = max(0, y)

            bbox = BoundingBox(x=x, y=y, width=w, height=h)

            # Extract landmarks
            keypoints = detection.get("keypoints", {})
            landmarks = Landmarks(
                left_eye=keypoints.get("left_eye"),
                right_eye=keypoints.get("right_eye"),
                nose=keypoints.get("nose"),
                left_mouth=keypoints.get("mouth_left"),
                right_mouth=keypoints.get("mouth_right"),
            )

            # Crop face region
            face_image = image[y : y + h, x : x + w].copy()

            results.append(
                DetectionResult(
                    bounding_box=bbox,
                    confidence=confidence,
                    landmarks=landmarks,
                    face_image=face_image,
                )
            )

        return results
