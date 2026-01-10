"""Custom exception classes for FaceVerify."""


class FaceVerifyError(Exception):
    """Base exception for FaceVerify errors."""

    pass


class NoFaceDetectedError(FaceVerifyError):
    """Raised when no face is detected in an image."""

    def __init__(self, message: str = "No face detected in the image"):
        self.message = message
        super().__init__(self.message)


class MultipleFacesError(FaceVerifyError):
    """Raised when multiple faces are detected but only one expected."""

    def __init__(
        self,
        message: str = "Multiple faces detected",
        face_count: int = 0,
    ):
        self.message = message
        self.face_count = face_count
        super().__init__(f"{self.message}: found {face_count} faces")


class InvalidImageError(FaceVerifyError):
    """Raised when the input image is invalid."""

    def __init__(self, message: str = "Invalid image"):
        self.message = message
        super().__init__(self.message)


class ModelNotFoundError(FaceVerifyError):
    """Raised when a required model cannot be found or loaded."""

    def __init__(
        self,
        model_name: str,
        message: str = "Model not found",
    ):
        self.model_name = model_name
        self.message = f"{message}: {model_name}"
        super().__init__(self.message)


class ConfigurationError(FaceVerifyError):
    """Raised when there is a configuration error."""

    def __init__(self, message: str = "Invalid configuration"):
        self.message = message
        super().__init__(self.message)
