"""Adaptive threshold decision maker."""

from typing import Tuple, List, Optional
import numpy as np
from faceverify.decision.base import BaseDecisionMaker


class AdaptiveDecisionMaker(BaseDecisionMaker):
    """
    Adaptive threshold-based verification decision.

    Adjusts threshold based on historical data and
    image quality factors.
    """

    def __init__(
        self,
        base_threshold: float = 0.65,
        min_threshold: float = 0.50,
        max_threshold: float = 0.80,
    ):
        """
        Initialize adaptive decision maker.

        Args:
            base_threshold: Starting threshold
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
        """
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self._history: List[Tuple[float, bool]] = []
        self._current_threshold = base_threshold

    def decide(
        self,
        similarity: float,
        distance: float,
        quality_score: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        Make adaptive threshold decision.

        Args:
            similarity: Similarity score
            distance: Distance between embeddings
            quality_score: Optional quality score (0 to 1)

        Returns:
            Tuple of (verified, confidence)
        """
        # Adjust threshold based on quality
        threshold = self._current_threshold

        if quality_score is not None:
            # Lower threshold for high-quality images
            quality_adjustment = (quality_score - 0.5) * 0.1
            threshold = float(
                np.clip(
                    threshold - quality_adjustment,
                    self.min_threshold,
                    self.max_threshold,
                )
            )

        verified = bool(similarity >= threshold)

        # Calculate confidence
        if verified:
            margin = similarity - threshold
            confidence = min(1.0, 0.5 + margin * 2)
        else:
            margin = threshold - similarity
            confidence = max(0.0, 0.5 - margin * 2)

        return verified, confidence

    def update(self, similarity: float, ground_truth: bool) -> None:
        """
        Update threshold based on feedback.

        Args:
            similarity: Similarity score from verification
            ground_truth: Actual label (True = same person)
        """
        self._history.append((similarity, ground_truth))

        # Adjust threshold if we have enough history
        if len(self._history) >= 10:
            self._optimize_threshold()

    def _optimize_threshold(self) -> None:
        """Optimize threshold based on history."""
        # Simple optimization: find threshold that minimizes errors
        best_threshold = self.base_threshold
        min_errors = float("inf")

        for t in np.arange(self.min_threshold, self.max_threshold, 0.01):
            errors = sum(1 for sim, gt in self._history if (sim >= t) != gt)
            if errors < min_errors:
                min_errors = errors
                best_threshold = t

        self._current_threshold = best_threshold
