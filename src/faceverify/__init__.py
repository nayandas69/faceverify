"""
FaceVerify - A production-ready face verification SDK.

This module provides a complete pipeline for face verification:
    Face Detection → Embedding Generation → Similarity Engine → Decision Engine

Example:
    >>> from faceverify import FaceVerifier
    >>> verifier = FaceVerifier()
    >>> result = verifier.verify("person1.jpg", "person2.jpg")
    >>> print(f"Verified: {result.verified}, Confidence: {result.confidence:.2%}")
"""

from faceverify.core.verifier import FaceVerifier
from faceverify.core.result import (
    VerificationResult,
    DetectionResult,
    EmbeddingResult,
    IdentificationResult,
)
from faceverify.config.settings import VerifierConfig
from faceverify.exceptions.errors import (
    FaceVerifyError,
    NoFaceDetectedError,
    MultipleFacesError,
    InvalidImageError,
    ModelNotFoundError,
)

__version__ = "1.0.0rc1"
__author__ = "nayandas69"
__email__ = "nayanchandradas@hotmail.com"
__license__ = "MIT"

__all__ = [
    # Main classes
    "FaceVerifier",
    "VerifierConfig",
    # Result types
    "VerificationResult",
    "DetectionResult",
    "EmbeddingResult",
    "IdentificationResult",
    # Exceptions
    "FaceVerifyError",
    "NoFaceDetectedError",
    "MultipleFacesError",
    "InvalidImageError",
    "ModelNotFoundError",
    # Metadata
    "__version__",
    "__author__",
]
