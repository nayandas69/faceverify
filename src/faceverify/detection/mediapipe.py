"""MediaPipe face detector implementation."""

from typing import List
import numpy as np

from faceverify.detection.base import BaseDetector
from faceverify.core.result import DetectionResult, BoundingBox, Landmarks


class MediaPipeDetector(BaseDetector):
    """
    MediaPipe Face Detection.

    Fast and efficient face detection using Google's MediaPipe.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        model_selection: int = 1,
        **kwargs,
    ):
        """
        Initialize MediaPipe detector.

        Args:
            confidence_threshold: Minimum detection confidence
            model_selection: 0 for short-range, 1 for full-range detection
        """
        super().__init__(confidence_threshold=confidence_threshold)
        self.model_selection = model_selection

    @property
    def name(self) -> str:
        return "mediapipe"

    def _load_model(self) -> None:
        """Load MediaPipe model."""
        try:
            import mediapipe as mp

            self._mp_face_detection = mp.solutions.face_detection
            self._model = self._mp_face_detection.FaceDetection(
                min_detection_confidence=self.confidence_threshold,
                model_selection=self.model_selection,
            )
        except ImportError:
            raise ImportError(
                "MediaPipe not installed. Install with: pip install mediapipe"
            )

    def _detect_impl(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect faces using MediaPipe."""
        results = []

        # Process image
        mp_results = self._model.process(image)

        if not mp_results.detections:
            return results

        h, w = image.shape[:2]

        for detection in mp_results.detections:
            confidence = detection.score[0]

            # Get bounding box (relative coordinates)
            bbox_rel = detection.location_data.relative_bounding_box

            x = int(bbox_rel.xmin * w)
            y = int(bbox_rel.ymin * h)
            width = int(bbox_rel.width * w)
            height = int(bbox_rel.height * h)

            # Clamp to image bounds
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)

            bbox = BoundingBox(x=x, y=y, width=width, height=height)

            # Extract keypoints
            keypoints = detection.location_data.relative_keypoints
            landmarks = None

            if len(keypoints) >= 6:
                landmarks = Landmarks(
                    right_eye=(keypoints[0].x * w, keypoints[0].y * h),
                    left_eye=(keypoints[1].x * w, keypoints[1].y * h),
                    nose=(keypoints[2].x * w, keypoints[2].y * h),
                    right_mouth=(keypoints[3].x * w, keypoints[3].y * h),
                    left_mouth=(keypoints[4].x * w, keypoints[4].y * h),
                )

            # Crop face
            face_image = image[y : y + height, x : x + width].copy()

            results.append(
                DetectionResult(
                    bounding_box=bbox,
                    confidence=confidence,
                    landmarks=landmarks,
                    face_image=face_image,
                )
            )

        return results
