"""Abstract base class for face embedding models."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseEmbedder(ABC):
    """
    Abstract base class for face embedding extraction.

    Face embeddings are dense vector representations of faces
    that can be compared to determine similarity.
    """

    def __init__(
        self,
        normalize: bool = True,
        **kwargs,
    ):
        """
        Initialize embedder.

        Args:
            normalize: Whether to L2-normalize embeddings
            **kwargs: Model-specific parameters
        """
        self.normalize = normalize
        self._model = None

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the dimension of output embeddings."""
        pass

    @property
    @abstractmethod
    def input_size(self) -> tuple:
        """Return expected input image size (height, width)."""
        pass

    @abstractmethod
    def _load_model(self) -> None:
        """Load the embedding model."""
        pass

    @abstractmethod
    def _extract_impl(self, face: np.ndarray) -> np.ndarray:
        """
        Implementation of embedding extraction.

        Args:
            face: Preprocessed face image

        Returns:
            Embedding vector
        """
        pass

    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for the model.

        Args:
            face: Input face image (RGB)

        Returns:
            Preprocessed image ready for the model
        """
        import cv2

        # Resize to expected input size
        target_size = self.input_size
        face = cv2.resize(face, (target_size[1], target_size[0]))

        # Convert to float and normalize
        face = face.astype(np.float32)

        # Model-specific normalization is done in subclasses
        return face

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """L2-normalize the embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def extract(self, face: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from a face image.

        Args:
            face: Face image (RGB, any size)

        Returns:
            Embedding vector of shape (embedding_dimension,)
        """
        if self._model is None:
            self._load_model()

        # Preprocess
        processed = self._preprocess(face)

        # Extract embedding
        embedding = self._extract_impl(processed)

        # Normalize if requested
        if self.normalize:
            embedding = self._normalize_embedding(embedding)

        return embedding

    def extract_batch(self, faces: list) -> np.ndarray:
        """
        Extract embeddings from multiple faces.

        Args:
            faces: List of face images

        Returns:
            Array of embeddings, shape (n_faces, embedding_dimension)
        """
        embeddings = []
        for face in faces:
            embedding = self.extract(face)
            embeddings.append(embedding)
        return np.array(embeddings)
