"""Core module containing the main verification pipeline."""

from faceverify.core.verifier import FaceVerifier
from faceverify.core.pipeline import VerificationPipeline
from faceverify.core.result import (
    VerificationResult,
    DetectionResult,
    EmbeddingResult,
    IdentificationResult,
)

__all__ = [
    "FaceVerifier",
    "VerificationPipeline",
    "VerificationResult",
    "DetectionResult",
    "EmbeddingResult",
    "IdentificationResult",
]
