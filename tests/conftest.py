"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import cv2


@pytest.fixture
def sample_face_image():
    """Generate a sample face-like image for testing."""
    # Create a simple image with face-like features
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    # Face oval (skin tone)
    cv2.ellipse(image, (100, 100), (70, 90), 0, 0, 360, (180, 160, 140), -1)

    # Eyes
    cv2.circle(image, (70, 80), 10, (50, 50, 50), -1)  # Left eye
    cv2.circle(image, (130, 80), 10, (50, 50, 50), -1)  # Right eye

    # Nose
    cv2.line(image, (100, 90), (100, 120), (150, 130, 110), 2)

    # Mouth
    cv2.ellipse(image, (100, 140), (25, 10), 0, 0, 180, (150, 100, 100), 2)

    return image


@pytest.fixture
def sample_face_image_2():
    """Generate a slightly different sample face image."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    # Face oval (different skin tone)
    cv2.ellipse(image, (100, 100), (65, 85), 0, 0, 360, (170, 150, 130), -1)

    # Eyes (slightly different positions)
    cv2.circle(image, (72, 78), 10, (45, 45, 45), -1)
    cv2.circle(image, (128, 78), 10, (45, 45, 45), -1)

    # Nose
    cv2.line(image, (100, 88), (100, 118), (145, 125, 105), 2)

    # Mouth
    cv2.ellipse(image, (100, 138), (22, 8), 0, 0, 180, (145, 95, 95), 2)

    return image


@pytest.fixture
def temp_image_file(sample_face_image):
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        cv2.imwrite(f.name, cv2.cvtColor(sample_face_image, cv2.COLOR_RGB2BGR))
        yield f.name

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def sample_embedding():
    """Generate a sample embedding vector."""
    np.random.seed(42)
    embedding = np.random.randn(512).astype(np.float32)
    # Normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


@pytest.fixture
def sample_embedding_similar():
    """Generate an embedding similar to sample_embedding."""
    np.random.seed(42)
    embedding = np.random.randn(512).astype(np.float32)
    # Add small noise
    noise = np.random.randn(512).astype(np.float32) * 0.1
    embedding = embedding + noise
    # Normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


@pytest.fixture
def sample_embedding_different():
    """Generate a clearly different embedding."""
    np.random.seed(123)  # Different seed
    embedding = np.random.randn(512).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding
