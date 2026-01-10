"""Main FaceVerifier class providing the public API."""

from typing import Optional, List, Tuple, Union
from pathlib import Path
import numpy as np
import structlog

from faceverify.config.settings import VerifierConfig
from faceverify.core.pipeline import VerificationPipeline
from faceverify.core.result import (
    VerificationResult,
    DetectionResult,
    EmbeddingResult,
    IdentificationResult,
)
from faceverify.utils.image import ImageInput

logger = structlog.get_logger(__name__)


class FaceVerifier:
    """
    Main class for face verification operations.

    This class provides a high-level API for:
        - Face verification (1:1 matching)
        - Face identification (1:N matching)
        - Face detection
        - Embedding extraction

    Example:
        >>> verifier = FaceVerifier()
        >>> result = verifier.verify("person1.jpg", "person2.jpg")
        >>> print(f"Same person: {result.verified}")

    Example with custom config:
        >>> config = VerifierConfig(
        ...     detector_backend="retinaface",
        ...     embedding_model="arcface",
        ...     threshold=0.65
        ... )
        >>> verifier = FaceVerifier(config)
    """

    def __init__(self, config: Optional[VerifierConfig] = None):
        """
        Initialize FaceVerifier.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or VerifierConfig()
        self._pipeline = VerificationPipeline(self.config)
        self._logger = logger.bind(component="verifier")

        self._logger.info(
            "FaceVerifier initialized",
            detector=self.config.detector_backend,
            model=self.config.embedding_model,
        )

    def verify(
        self,
        image1: ImageInput,
        image2: ImageInput,
    ) -> VerificationResult:
        """
        Verify if two images contain the same person.

        Args:
            image1: First image (path, URL, numpy array, or PIL Image)
            image2: Second image

        Returns:
            VerificationResult with verification outcome

        Example:
            >>> result = verifier.verify("img1.jpg", "img2.jpg")
            >>> if result.verified:
            ...     print(f"Match! Confidence: {result.confidence:.2%}")
        """
        return self._pipeline.verify(image1, image2)

    def verify_batch(
        self,
        pairs: List[Tuple[ImageInput, ImageInput]],
        parallel: bool = True,
    ) -> List[VerificationResult]:
        """
        Verify multiple image pairs.

        Args:
            pairs: List of (image1, image2) tuples to verify
            parallel: Whether to process in parallel (default: True)

        Returns:
            List of VerificationResults

        Example:
            >>> pairs = [("a1.jpg", "a2.jpg"), ("b1.jpg", "b2.jpg")]
            >>> results = verifier.verify_batch(pairs)
        """
        results = []

        for image1, image2 in pairs:
            try:
                result = self.verify(image1, image2)
                results.append(result)
            except Exception as e:
                self._logger.error(f"Batch verification error: {e}")
                # Create a failed result
                results.append(
                    VerificationResult(
                        verified=False,
                        confidence=0.0,
                        similarity=0.0,
                        distance=float("inf"),
                        threshold=self.config.threshold,
                        detector_backend=self.config.detector_backend,
                        embedding_model=self.config.embedding_model,
                        similarity_metric=self.config.similarity_metric,
                        processing_time=0.0,
                        metadata={"error": str(e)},
                    )
                )

        return results

    def identify(
        self,
        query_image: ImageInput,
        database: Union[str, Path, List[ImageInput]],
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> IdentificationResult:
        """
        Identify a face against a database of known faces (1:N matching).

        Args:
            query_image: Query image to identify
            database: Directory path or list of reference images
            top_k: Number of top matches to return
            threshold: Minimum similarity threshold (uses config default if None)

        Returns:
            IdentificationResult with matches

        Example:
            >>> result = verifier.identify("unknown.jpg", "./known_faces/")
            >>> if result.best_match:
            ...     print(f"Best match: {result.best_match['identity']}")
        """
        import time

        start_time = time.time()

        threshold = threshold or self.config.threshold

        # Extract query embedding
        query_embedding_result = self.extract_embedding(query_image)
        query_embedding = query_embedding_result.embedding

        # Get database images
        if isinstance(database, (str, Path)):
            database_path = Path(database)
            if not database_path.exists():
                raise ValueError(f"Database path does not exist: {database}")

            # Find all image files
            extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            database_images = [
                f for f in database_path.rglob("*") if f.suffix.lower() in extensions
            ]
        else:
            database_images = database

        matches = []

        for db_image in database_images:
            try:
                db_embedding_result = self.extract_embedding(db_image)
                similarity, distance = self._pipeline.compute_similarity(
                    query_embedding,
                    db_embedding_result.embedding,
                )

                if similarity >= threshold:
                    matches.append(
                        {
                            "identity": str(db_image),
                            "similarity": similarity,
                            "distance": distance,
                        }
                    )
            except Exception as e:
                self._logger.warning(f"Failed to process {db_image}: {e}")
                continue

        # Sort by similarity (descending)
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        matches = matches[:top_k]

        processing_time = time.time() - start_time

        return IdentificationResult(
            query_image=str(query_image),
            matches=matches,
            best_match=matches[0] if matches else None,
            total_candidates=len(database_images),
            processing_time=processing_time,
        )

    def detect_faces(
        self,
        image: ImageInput,
        return_all: bool = False,
    ) -> Union[DetectionResult, List[DetectionResult]]:
        """
        Detect faces in an image.

        Args:
            image: Input image
            return_all: If True, return all detected faces

        Returns:
            DetectionResult or list of DetectionResults
        """
        return self._pipeline.detect_faces(image, return_largest=not return_all)

    def extract_embedding(
        self,
        image: ImageInput,
    ) -> EmbeddingResult:
        """
        Extract face embedding from an image.

        Args:
            image: Input image containing a face

        Returns:
            EmbeddingResult with the face embedding vector
        """
        return self._pipeline.extract_embedding(image)

    def compare_embeddings(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compare two face embeddings directly.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Tuple of (similarity, distance)
        """
        return self._pipeline.compute_similarity(embedding1, embedding2)

    @property
    def threshold(self) -> float:
        """Current verification threshold."""
        return self.config.threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Update verification threshold."""
        if not 0 <= value <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.config.threshold = value
        if self._pipeline._decision_maker:
            self._pipeline._decision_maker.threshold = value
