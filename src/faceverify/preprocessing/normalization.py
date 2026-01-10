"""Image normalization utilities."""

from typing import Tuple, Optional
import numpy as np
import cv2


class ImageNormalizer:
    """
    Normalizes images for face recognition models.

    Handles color space conversion, histogram equalization,
    and intensity normalization.
    """

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        normalize_intensity: bool = True,
        equalize_histogram: bool = False,
    ):
        """
        Initialize normalizer.

        Args:
            target_size: Optional target size (height, width)
            normalize_intensity: Whether to normalize pixel intensities
            equalize_histogram: Whether to apply histogram equalization
        """
        self.target_size = target_size
        self.normalize_intensity = normalize_intensity
        self.equalize_histogram = equalize_histogram

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize an image.

        Args:
            image: Input image (RGB)

        Returns:
            Normalized image
        """
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Resize if needed
        if self.target_size is not None:
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]))

        # Histogram equalization
        if self.equalize_histogram:
            # Convert to LAB
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # Merge back
            lab = cv2.merge([l, a, b])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Intensity normalization
        if self.normalize_intensity:
            image = image.astype(np.float32)
            image = (image - image.mean()) / (image.std() + 1e-7)
            image = (image - image.min()) / (image.max() - image.min() + 1e-7) * 255
            image = image.astype(np.uint8)

        return image
