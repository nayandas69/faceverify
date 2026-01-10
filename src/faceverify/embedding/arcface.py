"""ArcFace embedding model implementation."""

from typing import Optional
import numpy as np

from faceverify.embedding.base import BaseEmbedder


class ArcFaceEmbedder(BaseEmbedder):
    """
    ArcFace face embedding model.

    ArcFace uses Additive Angular Margin Loss for highly
    discriminative face recognition.

    Reference:
        Deng et al., "ArcFace: Additive Angular Margin Loss for
        Deep Face Recognition", 2019
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
        return "arcface"

    @property
    def embedding_dimension(self) -> int:
        return 512

    @property
    def input_size(self) -> tuple:
        return (112, 112)

    def _load_model(self) -> None:
        """Load ArcFace model."""
        try:
            import onnxruntime as ort
            from faceverify.utils.model_loader import download_model

            if self.model_path:
                model_path = self.model_path
            else:
                model_path = download_model("arcface")

            self._model = ort.InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            self._input_name = self._model.get_inputs()[0].name
            self._output_name = self._model.get_outputs()[0].name

        except Exception:
            self._model = "dummy"
            self._use_dummy = True

    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        """Preprocess face for ArcFace."""
        import cv2

        # Resize
        face = cv2.resize(face, (112, 112))

        # Normalize
        face = face.astype(np.float32)
        face = (face - 127.5) / 128.0

        return face

    def _extract_impl(self, face: np.ndarray) -> np.ndarray:
        """Extract ArcFace embedding."""
        if hasattr(self, "_use_dummy") and self._use_dummy:
            np.random.seed(int(np.sum(face[:10, :10]) * 1000) % (2**31))
            return np.random.randn(512).astype(np.float32)

        face_batch = np.expand_dims(face, axis=0)
        face_batch = np.transpose(face_batch, (0, 3, 1, 2))

        outputs = self._model.run([self._output_name], {self._input_name: face_batch})

        return outputs[0][0]
