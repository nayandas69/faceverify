"""Configuration settings for FaceVerify."""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import os
import yaml

from faceverify.config.defaults import (
    DEFAULT_DETECTOR,
    DEFAULT_DETECTOR_CONFIDENCE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_NORMALIZE_EMBEDDINGS,
    DEFAULT_SIMILARITY_METRIC,
    DEFAULT_THRESHOLD,
    DEFAULT_FACE_SIZE,
    DEFAULT_ENABLE_GPU,
    DEFAULT_BATCH_SIZE,
    SUPPORTED_DETECTORS,
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_SIMILARITY_METRICS,
    get_default_threshold,
)


@dataclass
class VerifierConfig:
    """
    Configuration for FaceVerifier.

    Configuration can be loaded from:
        1. Constructor arguments
        2. YAML/JSON configuration file
        3. Environment variables (prefixed with FACEVERIFY_)

    Environment variables take precedence over file config,
    and constructor arguments take precedence over environment variables.

    Attributes:
        detector_backend: Face detection backend
        detector_confidence: Minimum confidence for face detection
        embedding_model: Model for generating face embeddings
        normalize_embeddings: Whether to L2-normalize embeddings
        similarity_metric: Metric for comparing embeddings
        threshold: Verification threshold
        face_size: Target face size for preprocessing
        enable_gpu: Whether to use GPU acceleration
        batch_size: Batch size for processing
    """

    # Detection settings
    detector_backend: str = DEFAULT_DETECTOR
    detector_confidence: float = DEFAULT_DETECTOR_CONFIDENCE

    # Embedding settings
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    normalize_embeddings: bool = DEFAULT_NORMALIZE_EMBEDDINGS

    # Similarity settings
    similarity_metric: str = DEFAULT_SIMILARITY_METRIC

    # Decision settings
    threshold: Optional[float] = None

    # Preprocessing settings
    face_size: Tuple[int, int] = DEFAULT_FACE_SIZE

    # Performance settings
    enable_gpu: bool = DEFAULT_ENABLE_GPU
    batch_size: int = DEFAULT_BATCH_SIZE

    # Extra metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and finalize configuration."""
        self._load_from_env()
        self._validate()

        # Set default threshold based on model/metric if not specified
        if self.threshold is None:
            self.threshold = get_default_threshold(
                self.embedding_model,
                self.similarity_metric,
            )

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            "FACEVERIFY_DETECTOR": "detector_backend",
            "FACEVERIFY_DETECTOR_CONFIDENCE": ("detector_confidence", float),
            "FACEVERIFY_EMBEDDING_MODEL": "embedding_model",
            "FACEVERIFY_NORMALIZE_EMBEDDINGS": (
                "normalize_embeddings",
                lambda x: x.lower() == "true",
            ),
            "FACEVERIFY_SIMILARITY_METRIC": "similarity_metric",
            "FACEVERIFY_THRESHOLD": ("threshold", float),
            "FACEVERIFY_ENABLE_GPU": ("enable_gpu", lambda x: x.lower() == "true"),
            "FACEVERIFY_BATCH_SIZE": ("batch_size", int),
        }

        for env_var, mapping in env_mappings.items():
            value = os.environ.get(env_var)
            if value is None:
                continue

            if isinstance(mapping, str):
                setattr(self, mapping, value)
            else:
                attr_name, converter = mapping
                setattr(self, attr_name, converter(value))

    def _validate(self) -> None:
        """Validate configuration values."""
        if self.detector_backend not in SUPPORTED_DETECTORS:
            raise ValueError(
                f"Invalid detector: {self.detector_backend}. "
                f"Supported: {SUPPORTED_DETECTORS}"
            )

        if self.embedding_model not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(
                f"Invalid embedding model: {self.embedding_model}. "
                f"Supported: {SUPPORTED_EMBEDDING_MODELS}"
            )

        if self.similarity_metric not in SUPPORTED_SIMILARITY_METRICS:
            raise ValueError(
                f"Invalid similarity metric: {self.similarity_metric}. "
                f"Supported: {SUPPORTED_SIMILARITY_METRICS}"
            )

        if not 0 <= self.detector_confidence <= 1:
            raise ValueError("detector_confidence must be between 0 and 1")

        if self.threshold is not None and not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VerifierConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        # Flatten nested config
        flat_config = {}

        if "detector" in data:
            flat_config["detector_backend"] = data["detector"].get("backend")
            flat_config["detector_confidence"] = data["detector"].get(
                "confidence_threshold"
            )

        if "embedding" in data:
            flat_config["embedding_model"] = data["embedding"].get("model")
            flat_config["normalize_embeddings"] = data["embedding"].get("normalize")

        if "similarity" in data:
            flat_config["similarity_metric"] = data["similarity"].get("metric")

        if "decision" in data:
            flat_config["threshold"] = data["decision"].get("threshold")

        if "performance" in data:
            flat_config["enable_gpu"] = data["performance"].get("enable_gpu")
            flat_config["batch_size"] = data["performance"].get("batch_size")

        # Remove None values
        flat_config = {k: v for k, v in flat_config.items() if v is not None}

        return cls(**flat_config)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "detector": {
                "backend": self.detector_backend,
                "confidence_threshold": self.detector_confidence,
            },
            "embedding": {
                "model": self.embedding_model,
                "normalize": self.normalize_embeddings,
            },
            "similarity": {
                "metric": self.similarity_metric,
            },
            "decision": {
                "threshold": self.threshold,
            },
            "performance": {
                "enable_gpu": self.enable_gpu,
                "batch_size": self.batch_size,
            },
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "detector_backend": self.detector_backend,
            "detector_confidence": self.detector_confidence,
            "embedding_model": self.embedding_model,
            "normalize_embeddings": self.normalize_embeddings,
            "similarity_metric": self.similarity_metric,
            "threshold": self.threshold,
            "face_size": self.face_size,
            "enable_gpu": self.enable_gpu,
            "batch_size": self.batch_size,
        }
