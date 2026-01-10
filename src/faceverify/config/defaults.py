"""Default configuration values for FaceVerify."""

from typing import Dict, Any

# Detection defaults
DEFAULT_DETECTOR = "mtcnn"
DEFAULT_DETECTOR_CONFIDENCE = 0.9
SUPPORTED_DETECTORS = ["mtcnn", "retinaface", "mediapipe", "opencv"]

# Embedding defaults
DEFAULT_EMBEDDING_MODEL = "facenet"
DEFAULT_EMBEDDING_DIMENSION = 512
DEFAULT_NORMALIZE_EMBEDDINGS = True
SUPPORTED_EMBEDDING_MODELS = ["facenet", "arcface", "vggface"]

# Similarity defaults
DEFAULT_SIMILARITY_METRIC = "cosine"
SUPPORTED_SIMILARITY_METRICS = ["cosine", "euclidean", "manhattan"]

# Decision defaults
DEFAULT_THRESHOLD = 0.65

# Preprocessing defaults
DEFAULT_FACE_SIZE = (160, 160)

# Performance defaults
DEFAULT_ENABLE_GPU = True
DEFAULT_BATCH_SIZE = 32

# Model-specific thresholds (optimized values)
MODEL_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "facenet": {
        "cosine": 0.65,
        "euclidean": 0.55,
    },
    "arcface": {
        "cosine": 0.68,
        "euclidean": 0.52,
    },
    "vggface": {
        "cosine": 0.60,
        "euclidean": 0.58,
    },
}


def get_default_threshold(
    model: str = DEFAULT_EMBEDDING_MODEL,
    metric: str = DEFAULT_SIMILARITY_METRIC,
) -> float:
    """Get optimized threshold for model/metric combination."""
    return MODEL_THRESHOLDS.get(model, {}).get(metric, DEFAULT_THRESHOLD)
