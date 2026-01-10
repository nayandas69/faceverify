"""Verification pipeline orchestrating the face verification process."""

from typing import Optional, List, Tuple, Union
from pathlib import Path
import time
import numpy as np
import structlog

from faceverify.detection.base import BaseDetector
from faceverify.detection.factory import create_detector
from faceverify.embedding.base import BaseEmbedder
from faceverify.embedding.factory import create_embedder
from faceverify.similarity.engine import SimilarityEngine
from faceverify.decision.base import BaseDecisionMaker
from faceverify.decision.threshold import ThresholdDecisionMaker
from faceverify.preprocessing.alignment import FaceAligner
from faceverify.preprocessing.normalization import ImageNormalizer
from faceverify.config.settings import VerifierConfig
from faceverify.core.result import (
    VerificationResult,
    DetectionResult,
    EmbeddingResult,
)
from faceverify.utils.image import load_image, ImageInput
from faceverify.exceptions.errors import NoFaceDetectedError, InvalidImageError

logger = structlog.get_logger(__name__)


class VerificationPipeline:
    """
    Orchestrates the complete face verification pipeline.

    Pipeline stages:
        1. Face Detection - Locate faces in images
        2. Preprocessing - Align and normalize faces
        3. Embedding Generation - Extract face embeddings
        4. Similarity Computation - Calculate similarity between embeddings
        5. Decision Making - Determine verification result
    """

    def __init__(self, config: VerifierConfig):
        """
        Initialize the verification pipeline.

        Args:
            config: Configuration settings for the pipeline
        """
        self.config = config
        self._logger = logger.bind(component="pipeline")

        # Initialize pipeline components
        self._detector: Optional[BaseDetector] = None
        self._embedder: Optional[BaseEmbedder] = None
        self._similarity_engine: Optional[SimilarityEngine] = None
        self._decision_maker: Optional[BaseDecisionMaker] = None
        self._aligner: Optional[FaceAligner] = None
        self._normalizer: Optional[ImageNormalizer] = None

        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of pipeline components."""
        if self._initialized:
            return

        self._logger.info("Initializing pipeline components")
        start_time = time.time()

        # Create detector
        self._detector = create_detector(
            backend=self.config.detector_backend,
            confidence_threshold=self.config.detector_confidence,
        )

        # Create embedder
        self._embedder = create_embedder(
            model_name=self.config.embedding_model,
            normalize=self.config.normalize_embeddings,
        )

        # Create similarity engine
        self._similarity_engine = SimilarityEngine(metric=self.config.similarity_metric)

        # Create decision maker
        self._decision_maker = ThresholdDecisionMaker(threshold=self.config.threshold)

        # Create preprocessors
        self._aligner = FaceAligner(
            output_size=self.config.face_size,
        )
        self._normalizer = ImageNormalizer()

        self._initialized = True
        init_time = time.time() - start_time
        self._logger.info("Pipeline initialized", init_time=f"{init_time:.2f}s")

    def detect_faces(
        self,
        image: ImageInput,
        return_largest: bool = True,
    ) -> Union[DetectionResult, List[DetectionResult]]:
        """
        Detect faces in an image.

        Args:
            image: Input image (path, URL, numpy array, or PIL Image)
            return_largest: If True, return only the largest face

        Returns:
            Detection result(s)

        Raises:
            NoFaceDetectedError: If no faces are found
            InvalidImageError: If the image cannot be loaded
        """
        self._ensure_initialized()

        # Load image
        img_array = load_image(image)
        if img_array is None:
            raise InvalidImageError(f"Failed to load image: {image}")

        # Detect faces
        detections = self._detector.detect(img_array)

        if not detections:
            raise NoFaceDetectedError("No face detected in the image")

        if return_largest:
            # Return the detection with the largest bounding box area
            return max(detections, key=lambda d: d.bounding_box.area)

        return detections

    def extract_embedding(
        self,
        image: ImageInput,
        detection: Optional[DetectionResult] = None,
    ) -> EmbeddingResult:
        """
        Extract face embedding from an image.

        Args:
            image: Input image
            detection: Optional pre-computed detection result

        Returns:
            Embedding result containing the face embedding vector
        """
        self._ensure_initialized()
        start_time = time.time()

        # Load image if needed
        img_array = load_image(image)

        # Detect face if not provided
        if detection is None:
            detection = self.detect_faces(img_array, return_largest=True)

        # Get face region
        if detection.aligned_face is not None:
            face_img = detection.aligned_face
        elif detection.face_image is not None:
            face_img = detection.face_image
        else:
            # Crop face from image
            bbox = detection.bounding_box
            face_img = img_array[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]

        # Align face if landmarks available
        if detection.landmarks is not None:
            face_img = self._aligner.align(face_img, detection.landmarks)

        # Normalize image
        face_img = self._normalizer.normalize(face_img)

        # Extract embedding
        embedding = self._embedder.extract(face_img)

        processing_time = time.time() - start_time

        return EmbeddingResult(
            embedding=embedding,
            model_name=self._embedder.model_name,
            dimension=embedding.shape[0],
            normalized=self.config.normalize_embeddings,
            processing_time=processing_time,
        )

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding

        Returns:
            Tuple of (similarity score, distance)
        """
        self._ensure_initialized()
        return self._similarity_engine.compute(embedding1, embedding2)

    def make_decision(
        self,
        similarity: float,
        distance: float,
    ) -> Tuple[bool, float]:
        """
        Make verification decision based on similarity.

        Args:
            similarity: Similarity score between faces
            distance: Distance between embeddings

        Returns:
            Tuple of (verified, confidence)
        """
        self._ensure_initialized()
        return self._decision_maker.decide(similarity, distance)

    def verify(
        self,
        image1: ImageInput,
        image2: ImageInput,
    ) -> VerificationResult:
        """
        Verify if two images contain the same person.

        This is the main entry point for face verification.

        Args:
            image1: First image
            image2: Second image

        Returns:
            VerificationResult containing the verification outcome
        """
        self._ensure_initialized()
        start_time = time.time()

        self._logger.debug("Starting verification")

        # Extract embeddings for both images
        embedding_result1 = self.extract_embedding(image1)
        embedding_result2 = self.extract_embedding(image2)

        # Compute similarity
        similarity, distance = self.compute_similarity(
            embedding_result1.embedding,
            embedding_result2.embedding,
        )

        # Make decision
        verified, confidence = self.make_decision(similarity, distance)

        processing_time = time.time() - start_time

        result = VerificationResult(
            verified=verified,
            confidence=confidence,
            similarity=similarity,
            distance=distance,
            threshold=self.config.threshold,
            detector_backend=self.config.detector_backend,
            embedding_model=self.config.embedding_model,
            similarity_metric=self.config.similarity_metric,
            processing_time=processing_time,
        )

        self._logger.info(
            "Verification complete",
            verified=verified,
            similarity=f"{similarity:.4f}",
            time=f"{processing_time:.3f}s",
        )

        return result
