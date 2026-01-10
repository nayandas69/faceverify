"""RetinaFace detector implementation."""

from typing import List
import numpy as np

from faceverify.detection.base import BaseDetector
from faceverify.core.result import DetectionResult, BoundingBox, Landmarks


class RetinaFaceDetector(BaseDetector):
    """
    RetinaFace detector - high accuracy face detection.

    RetinaFace provides excellent detection accuracy with
    facial landmark localization.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        **kwargs,
    ):
        super().__init__(confidence_threshold=confidence_threshold)
        self._retinaface = None

    @property
    def name(self) -> str:
        return "retinaface"

    def _load_model(self) -> None:
        """Load RetinaFace model."""
        try:
            from retinaface import RetinaFace

            self._retinaface = RetinaFace
            self._model = True  # RetinaFace uses static methods
        except ImportError:
            raise ImportError(
                "RetinaFace not installed. Install with: pip install retinaface"
            )

    def _detect_impl(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect faces using RetinaFace."""
        # RetinaFace.detect_faces returns a dict
        detections = self._retinaface.detect_faces(image)

        results = []

        for face_id, detection in detections.items():
            confidence = detection["score"]

            # Extract bounding box (x1, y1, x2, y2 format)
            facial_area = detection["facial_area"]
            x1, y1, x2, y2 = facial_area

            bbox = BoundingBox(
                x=int(x1),
                y=int(y1),
                width=int(x2 - x1),
                height=int(y2 - y1),
            )

            # Extract landmarks
            landmarks_dict = detection.get("landmarks", {})
            landmarks = Landmarks(
                left_eye=landmarks_dict.get("left_eye"),
                right_eye=landmarks_dict.get("right_eye"),
                nose=landmarks_dict.get("nose"),
                left_mouth=landmarks_dict.get("mouth_left"),
                right_mouth=landmarks_dict.get("mouth_right"),
            )

            # Crop face
            face_image = image[int(y1) : int(y2), int(x1) : int(x2)].copy()

            results.append(
                DetectionResult(
                    bounding_box=bbox,
                    confidence=confidence,
                    landmarks=landmarks,
                    face_image=face_image,
                )
            )

        return results
