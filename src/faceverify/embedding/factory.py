"""Factory for creating face embedding models."""

from faceverify.embedding.base import BaseEmbedder


def create_embedder(
    model_name: str = "facenet",
    normalize: bool = True,
    **kwargs,
) -> BaseEmbedder:
    """
    Create a face embedding model instance.

    Args:
        model_name: Name of the embedding model
        normalize: Whether to L2-normalize embeddings
        **kwargs: Model-specific parameters

    Returns:
        Configured embedder instance

    Raises:
        ValueError: If model is not supported
    """
    model_name = model_name.lower()

    if model_name == "facenet":
        from faceverify.embedding.facenet import FaceNetEmbedder

        return FaceNetEmbedder(normalize=normalize, **kwargs)

    elif model_name == "arcface":
        from faceverify.embedding.arcface import ArcFaceEmbedder

        return ArcFaceEmbedder(normalize=normalize, **kwargs)

    elif model_name == "vggface":
        from faceverify.embedding.vggface import VGGFaceEmbedder

        return VGGFaceEmbedder(normalize=normalize, **kwargs)

    else:
        raise ValueError(
            f"Unknown embedding model: {model_name}. "
            f"Supported: facenet, arcface, vggface"
        )
