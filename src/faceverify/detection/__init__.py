"""Face detection module with multiple backend support."""

from faceverify.detection.base import BaseDetector
from faceverify.detection.factory import create_detector
from faceverify.detection.mtcnn import MTCNNDetector
from faceverify.detection.opencv import OpenCVDetector

__all__ = [
    "BaseDetector",
    "create_detector",
    "MTCNNDetector",
    "OpenCVDetector",
]
