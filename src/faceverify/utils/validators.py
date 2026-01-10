"""Input validation utilities."""

from typing import Any, Optional
import numpy as np

from faceverify.exceptions.errors import InvalidImageError


def validate_image(image: Any) -> bool:
    """
    Validate that input is a valid image.

    Args:
        image: Input to validate

    Returns:
        True if valid

    Raises:
        InvalidImageError: If validation fails
    """
    if image is None:
        raise InvalidImageError("Image cannot be None")

    if isinstance(image, np.ndarray):
        if len(image.shape) not in (2, 3):
            raise InvalidImageError(
                f"Invalid image shape: {image.shape}. "
                "Expected 2D (grayscale) or 3D (color) array."
            )

        if len(image.shape) == 3 and image.shape[2] not in (1, 3, 4):
            raise InvalidImageError(
                f"Invalid number of channels: {image.shape[2]}. "
                "Expected 1, 3, or 4 channels."
            )

        if image.size == 0:
            raise InvalidImageError("Image is empty")

    return True


def validate_embedding(
    embedding: Any,
    expected_dim: Optional[int] = None,
) -> bool:
    """
    Validate that input is a valid embedding vector.

    Args:
        embedding: Input to validate
        expected_dim: Expected dimension (optional)

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    if embedding is None:
        raise ValueError("Embedding cannot be None")

    if not isinstance(embedding, np.ndarray):
        raise ValueError(f"Embedding must be numpy array, got {type(embedding)}")

    if len(embedding.shape) != 1:
        raise ValueError(f"Embedding must be 1D array, got shape {embedding.shape}")

    if expected_dim is not None and embedding.shape[0] != expected_dim:
        raise ValueError(
            f"Expected embedding dimension {expected_dim}, " f"got {embedding.shape[0]}"
        )

    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains NaN or infinite values")

    return True
