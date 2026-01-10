"""GPU utilities and detection."""

from typing import List, Optional, Dict, Any
import os


def get_available_gpus() -> List[Dict[str, Any]]:
    """
    Get list of available GPUs.

    Returns:
        List of GPU information dictionaries
    """
    gpus = []

    # Try CUDA
    try:
        import tensorflow as tf

        cuda_gpus = tf.config.list_physical_devices("GPU")
        for i, gpu in enumerate(cuda_gpus):
            gpus.append(
                {
                    "id": i,
                    "name": gpu.name,
                    "type": "CUDA",
                }
            )
    except Exception:
        pass

    return gpus


def is_gpu_available() -> bool:
    """Check if GPU is available."""
    return len(get_available_gpus()) > 0


def set_gpu_memory_growth(enable: bool = True) -> None:
    """
    Enable or disable GPU memory growth.

    When enabled, TensorFlow allocates GPU memory as needed
    rather than all at once.

    Args:
        enable: Whether to enable memory growth
    """
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, enable)
    except Exception:
        pass


def set_visible_gpus(gpu_ids: Optional[List[int]] = None) -> None:
    """
    Set which GPUs are visible to the application.

    Args:
        gpu_ids: List of GPU IDs to use, or None for all
    """
    if gpu_ids is None:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
