"""Model downloading and caching utilities."""

from pathlib import Path
from typing import Optional
import os
import hashlib
import urllib.request
import structlog

logger = structlog.get_logger(__name__)

MODEL_REGISTRY = {
    "facenet": {
        "url": "https://huggingface.co/rocca/facenet-keras/resolve/main/facenet_keras.h5",
        "checksum": None,
        "filename": "facenet_keras.h5",
    },
    "arcface": {
        "url": "https://huggingface.co/rocca/arcface-keras/resolve/main/arcface_keras.h5",
        "checksum": None,
        "filename": "arcface_keras.h5",
    },
    "vggface": {
        "url": "https://huggingface.co/rocca/vggface-keras/resolve/main/vggface_keras.h5",
        "checksum": None,
        "filename": "vggface_keras.h5",
    },
}


def get_model_dir() -> Path:
    """Get the model cache directory."""
    # Check environment variable
    cache_dir = os.environ.get("FACEVERIFY_MODEL_DIR")

    if cache_dir:
        model_dir = Path(cache_dir)
    else:
        # Default to ~/.faceverify/models
        model_dir = Path.home() / ".faceverify" / "models"

    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def download_model(model_name: str) -> Optional[str]:
    """
    Download a model if not cached.

    Args:
        model_name: Name of the model to download

    Returns:
        Path to the model file, or None if download fails

    Raises:
        ValueError: If model is not in registry
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. " f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_info = MODEL_REGISTRY[model_name]
    model_dir = get_model_dir()
    model_path = model_dir / model_info["filename"]

    # Check if already downloaded
    if model_path.exists():
        # Verify checksum if available
        if model_info["checksum"]:
            if _verify_checksum(model_path, model_info["checksum"]):
                logger.debug(f"Model {model_name} found in cache")
                return str(model_path)
            else:
                logger.warning(f"Model {model_name} checksum mismatch, re-downloading")
        else:
            return str(model_path)

    # Download model
    logger.info(f"Downloading model: {model_name}")

    try:
        # Add headers to avoid 403 errors
        request = urllib.request.Request(
            model_info["url"], headers={"User-Agent": "FaceVerify/1.0"}
        )
        with urllib.request.urlopen(request) as response:
            with open(model_path, "wb") as f:
                f.write(response.read())
        logger.info(f"Model {model_name} downloaded successfully")
        return str(model_path)
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")
        logger.info(f"Using local feature extraction instead")
        return None  # Return None to trigger local fallback


def _verify_checksum(path: Path, expected: str) -> bool:
    """Verify file checksum."""
    sha256_hash = hashlib.sha256()

    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest() == expected
