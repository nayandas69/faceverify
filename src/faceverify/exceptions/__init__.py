"""Custom exceptions for FaceVerify."""

from faceverify.exceptions.errors import (
    FaceVerifyError,
    NoFaceDetectedError,
    MultipleFacesError,
    InvalidImageError,
    ModelNotFoundError,
    ConfigurationError,
)

__all__ = [
    "FaceVerifyError",
    "NoFaceDetectedError",
    "MultipleFacesError",
    "InvalidImageError",
    "ModelNotFoundError",
    "ConfigurationError",
]
