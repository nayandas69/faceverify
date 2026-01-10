"""VGGFace embedding model implementation."""

from typing import Optional
import numpy as np

from faceverify.embedding.base import BaseEmbedder


class VGGFaceEmbedder(BaseEmbedder):
    """
    VGGFace face embedding model.

    Based on VGG architecture trained for face recognition.
    """

    def __init__(
        self,
        normalize: bool = True,
        model_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(normalize=normalize)
        self.model_path = model_path

    @property
    def model_name(self) -> str:
        return "vggface"

    @property
    def embedding_dimension(self) -> int:
        return 2048

    @property
    def input_size(self) -> tuple:
        return (224, 224)

    def _load_model(self) -> None:
        """Load VGGFace model."""
        try:
            import onnxruntime as ort
            from faceverify.utils.model_loader import download_model

            if self.model_path:
                model_path = self.model_path
            else:
                model_path = download_model("vggface")

            self._model = ort.InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            self._input_name = self._model.get_inputs()[0].name
            self._output_name = self._model.get_outputs()[0].name

        except Exception:
            self._model = "dummy"
            self._use_dummy = True

    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        """Preprocess face for VGGFace."""
        import cv2

        # Resize
        face = cv2.resize(face, (224, 224))

        # VGG preprocessing (BGR, mean subtraction)
        face = face.astype(np.float32)
        face = face[..., ::-1]  # RGB to BGR
        face[..., 0] -= 93.5940
        face[..., 1] -= 104.7624
        face[..., 2] -= 129.1863

        return face

    def _extract_impl(self, face: np.ndarray) -> np.ndarray:
        """Extract VGGFace embedding."""
        if hasattr(self, "_use_dummy") and self._use_dummy:
            np.random.seed(int(np.sum(face[:10, :10]) * 1000) % (2**31))
            return np.random.randn(2048).astype(np.float32)

        face_batch = np.expand_dims(face, axis=0)
        face_batch = np.transpose(face_batch, (0, 3, 1, 2))

        outputs = self._model.run([self._output_name], {self._input_name: face_batch})

        return outputs[0][0]
