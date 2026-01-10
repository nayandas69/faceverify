"""Face alignment utilities."""

from typing import Tuple, Optional
import numpy as np
import cv2

from faceverify.core.result import Landmarks


class FaceAligner:
    """
    Aligns faces based on facial landmarks.

    Face alignment normalizes the position, scale, and rotation
    of faces to improve embedding quality.
    """

    # Standard landmark positions (relative to output size)
    STANDARD_LANDMARKS = {
        "left_eye": (0.35, 0.35),
        "right_eye": (0.65, 0.35),
        "nose": (0.5, 0.55),
        "left_mouth": (0.35, 0.75),
        "right_mouth": (0.65, 0.75),
    }

    def __init__(
        self,
        output_size: Tuple[int, int] = (160, 160),
        border_mode: int = cv2.BORDER_CONSTANT,
    ):
        """
        Initialize face aligner.

        Args:
            output_size: Target output size (height, width)
            border_mode: OpenCV border mode for transformation
        """
        self.output_size = output_size
        self.border_mode = border_mode

    def align(
        self,
        face: np.ndarray,
        landmarks: Optional[Landmarks] = None,
    ) -> np.ndarray:
        """
        Align a face image using landmarks.

        Args:
            face: Input face image
            landmarks: Facial landmarks (optional)

        Returns:
            Aligned face image
        """
        if landmarks is None:
            # If no landmarks, just resize
            return cv2.resize(face, (self.output_size[1], self.output_size[0]))

        # Get source and destination points
        src_points = []
        dst_points = []

        h, w = self.output_size

        for name in ["left_eye", "right_eye"]:
            src_pt = getattr(landmarks, name)
            if src_pt is not None:
                src_points.append(src_pt)
                dst_rel = self.STANDARD_LANDMARKS[name]
                dst_points.append((dst_rel[0] * w, dst_rel[1] * h))

        if len(src_points) < 2:
            # Not enough landmarks, just resize
            return cv2.resize(face, (self.output_size[1], self.output_size[0]))

        # Estimate transformation
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        # Use similarity transform (rotation, scale, translation)
        transform = cv2.estimateAffinePartial2D(src_points, dst_points)[0]

        if transform is None:
            return cv2.resize(face, (self.output_size[1], self.output_size[0]))

        # Apply transformation
        aligned = cv2.warpAffine(
            face,
            transform,
            (self.output_size[1], self.output_size[0]),
            borderMode=self.border_mode,
        )

        return aligned

    def align_by_eyes(
        self,
        face: np.ndarray,
        left_eye: Tuple[float, float],
        right_eye: Tuple[float, float],
    ) -> np.ndarray:
        """
        Align face using only eye coordinates.

        Args:
            face: Input face image
            left_eye: Left eye coordinates (x, y)
            right_eye: Right eye coordinates (x, y)

        Returns:
            Aligned face image
        """
        # Calculate rotation angle
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Calculate eye center
        eye_center = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2,
        )

        # Calculate scale based on eye distance
        eye_dist = np.sqrt(dx**2 + dy**2)
        target_eye_dist = self.output_size[1] * 0.3  # 30% of width
        scale = target_eye_dist / eye_dist

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale)

        # Adjust for output centering
        rotation_matrix[0, 2] += self.output_size[1] / 2 - eye_center[0]
        rotation_matrix[1, 2] += self.output_size[0] * 0.35 - eye_center[1]

        # Apply transformation
        aligned = cv2.warpAffine(
            face,
            rotation_matrix,
            (self.output_size[1], self.output_size[0]),
            borderMode=self.border_mode,
        )

        return aligned
