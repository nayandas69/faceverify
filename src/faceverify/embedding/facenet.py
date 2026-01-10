"""FaceNet embedding model implementation using DeepFace."""

from typing import Optional
import numpy as np
import os
import tempfile

from faceverify.embedding.base import BaseEmbedder


class FaceNetEmbedder(BaseEmbedder):
    """
    FaceNet face embedding model using DeepFace library.

    Uses pre-trained deep learning models for accurate face embeddings.
    No compilation required - works on Windows out of the box.

    Reference:
        Schroff et al., "FaceNet: A Unified Embedding for Face
        Recognition and Clustering", 2015
    """

    def __init__(
        self,
        normalize: bool = True,
        model_name: str = "Facenet512",
        **kwargs,
    ):
        """
        Initialize FaceNet embedder.

        Args:
            normalize: Whether to L2-normalize embeddings
            model_name: DeepFace model to use (Facenet512, Facenet, ArcFace, VGG-Face)
        """
        super().__init__(normalize=normalize)
        self._model_name = model_name
        self._deepface = None
        self._initialized = False

    @property
    def model_name(self) -> str:
        return "facenet"

    @property
    def embedding_dimension(self) -> int:
        return 512

    @property
    def input_size(self) -> tuple:
        return (160, 160)

    def _load_model(self) -> None:
        """Load DeepFace model."""
        try:
            from deepface import DeepFace

            self._deepface = DeepFace

            # Pre-load the model by doing a dummy embedding
            # This downloads the model weights on first run
            dummy = np.zeros((160, 160, 3), dtype=np.uint8)
            try:
                self._deepface.represent(
                    img_path=dummy,
                    model_name=self._model_name,
                    enforce_detection=False,
                    detector_backend="skip",
                )
            except Exception:
                pass  # Model is loaded even if dummy fails

            self._initialized = True

        except ImportError as e:
            raise ImportError(
                "DeepFace is required. Install with: pip install deepface tf-keras"
            ) from e

    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        """Preprocess face for embedding extraction."""
        import cv2

        if face is None or face.size == 0:
            raise ValueError("Empty face image")

        # Ensure correct format
        if len(face.shape) == 2:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        elif face.shape[2] == 4:
            face = cv2.cvtColor(face, cv2.COLOR_BGRA2RGB)
        elif face.shape[2] == 3:
            # OpenCV loads as BGR, convert to RGB
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Resize to standard size
        face = cv2.resize(face, (160, 160))

        return face.astype(np.uint8)

    def _extract_impl(self, face: np.ndarray) -> np.ndarray:
        """Extract face embedding using DeepFace."""
        import cv2

        if self._deepface is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Try direct numpy array approach
        try:
            result = self._deepface.represent(
                img_path=face,
                model_name=self._model_name,
                enforce_detection=False,
                detector_backend="skip",
            )

            if result and len(result) > 0:
                embedding = np.array(result[0]["embedding"], dtype=np.float32)
                return embedding

        except Exception:
            pass

        # Fallback: save to temp file and process
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                temp_path = f.name
                # Convert RGB to BGR for OpenCV save
                bgr_face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(temp_path, bgr_face)

            result = self._deepface.represent(
                img_path=temp_path,
                model_name=self._model_name,
                enforce_detection=False,
                detector_backend="skip",
            )

            # Clean up temp file
            os.unlink(temp_path)

            if result and len(result) > 0:
                embedding = np.array(result[0]["embedding"], dtype=np.float32)
                return embedding

        except Exception as e:
            # Clean up temp file on error
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise RuntimeError(f"Failed to extract embedding: {e}")

        raise RuntimeError("Failed to extract face embedding")
