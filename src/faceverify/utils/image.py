"""Image loading and processing utilities."""

from typing import Union, Optional
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import urllib.request
import io

# Type alias for image inputs
ImageInput = Union[str, Path, np.ndarray, Image.Image]


def load_image(source: ImageInput) -> Optional[np.ndarray]:
    """
    Load an image from various sources.

    Supports:
        - File path (string or Path)
        - URL (http/https)
        - NumPy array
        - PIL Image
        - Base64 encoded string

    Args:
        source: Image source

    Returns:
        Image as RGB numpy array, or None if loading fails
    """
    try:
        # Already a numpy array
        if isinstance(source, np.ndarray):
            return _ensure_rgb(source)

        # PIL Image
        if isinstance(source, Image.Image):
            return np.array(source.convert("RGB"))

        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # String - could be path, URL, or base64
        if isinstance(source, str):
            # URL
            if source.startswith(("http://", "https://")):
                return _load_from_url(source)

            # Base64
            if source.startswith("data:image"):
                return _load_from_base64(source)

            # File path
            return _load_from_file(source)

        return None

    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure image is in RGB format."""
    if len(image.shape) == 2:
        # Grayscale to RGB
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 4:
        # RGBA to RGB
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    if image.shape[2] == 3:
        # Could be BGR (from OpenCV) - check and convert if needed
        # For safety, we assume the input is already RGB
        return image

    return image


def _load_from_file(path: str) -> Optional[np.ndarray]:
    """Load image from file path."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    # Use OpenCV for loading
    image = cv2.imread(path)
    if image is None:
        # Try PIL as fallback
        pil_image = Image.open(path)
        return np.array(pil_image.convert("RGB"))

    # Convert BGR to RGB
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_from_url(url: str) -> Optional[np.ndarray]:
    """Load image from URL."""
    with urllib.request.urlopen(url, timeout=10) as response:
        image_data = response.read()

    # Decode image
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        # Try PIL as fallback
        pil_image = Image.open(io.BytesIO(image_data))
        return np.array(pil_image.convert("RGB"))

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_from_base64(data: str) -> Optional[np.ndarray]:
    """Load image from base64 string."""
    import base64

    # Remove data URL prefix if present
    if "," in data:
        data = data.split(",")[1]

    image_data = base64.b64decode(data)
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        pil_image = Image.open(io.BytesIO(image_data))
        return np.array(pil_image.convert("RGB"))

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, path: str) -> None:
    """
    Save image to file.

    Args:
        image: RGB image as numpy array
        path: Output file path
    """
    # Convert RGB to BGR for OpenCV
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr_image)
