"""Face embedding module with multiple model support."""

from faceverify.embedding.base import BaseEmbedder
from faceverify.embedding.factory import create_embedder

__all__ = [
    "BaseEmbedder",
    "create_embedder",
]
