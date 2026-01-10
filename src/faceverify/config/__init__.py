"""Configuration module for FaceVerify."""

from faceverify.config.settings import VerifierConfig
from faceverify.config.defaults import (
    DEFAULT_DETECTOR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SIMILARITY_METRIC,
    DEFAULT_THRESHOLD,
)

VerificationConfig = VerifierConfig

__all__ = [
    "VerifierConfig",
    "VerificationConfig",  # Alias
    "DEFAULT_DETECTOR",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_SIMILARITY_METRIC",
    "DEFAULT_THRESHOLD",
]
