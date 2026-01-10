"""Factory for creating face detectors."""

from typing import Optional
from faceverify.detection.base import BaseDetector


def create_detector(
    backend: str = "mtcnn",
    confidence_threshold: float = 0.9,
    **kwargs,
) -> BaseDetector:
    """
    Create a face detector instance.

    Args:
        backend: Detection backend name
        confidence_threshold: Minimum confidence for detections
        **kwargs: Backend-specific parameters

    Returns:
        Configured detector instance

    Raises:
        ValueError: If backend is not supported
    """
    backend = backend.lower()

    if backend == "mtcnn":
        from faceverify.detection.mtcnn import MTCNNDetector

        return MTCNNDetector(confidence_threshold=confidence_threshold, **kwargs)

    elif backend == "opencv":
        from faceverify.detection.opencv import OpenCVDetector

        return OpenCVDetector(confidence_threshold=confidence_threshold, **kwargs)

    elif backend == "retinaface":
        from faceverify.detection.retinaface import RetinaFaceDetector

        return RetinaFaceDetector(confidence_threshold=confidence_threshold, **kwargs)

    elif backend == "mediapipe":
        from faceverify.detection.mediapipe import MediaPipeDetector

        return MediaPipeDetector(confidence_threshold=confidence_threshold, **kwargs)

    else:
        raise ValueError(
            f"Unknown detector backend: {backend}. "
            f"Supported: mtcnn, opencv, retinaface, mediapipe"
        )
