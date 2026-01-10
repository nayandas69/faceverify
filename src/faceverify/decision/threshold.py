"""Threshold-based decision maker."""

from typing import Tuple
from faceverify.decision.base import BaseDecisionMaker


class ThresholdDecisionMaker(BaseDecisionMaker):
    """
    Simple threshold-based verification decision.

    Makes a binary decision based on whether the similarity
    score exceeds a predefined threshold.
    """

    def __init__(self, threshold: float = 0.65):
        """
        Initialize decision maker.

        Args:
            threshold: Similarity threshold for positive match
        """
        self.threshold = threshold

    def decide(
        self,
        similarity: float,
        distance: float,
    ) -> Tuple[bool, float]:
        """
        Make threshold-based decision.

        Args:
            similarity: Similarity score (0 to 1)
            distance: Distance between embeddings

        Returns:
            Tuple of (verified, confidence)
        """
        verified = similarity >= self.threshold

        # Calculate confidence based on margin from threshold
        if verified:
            # How much above threshold
            margin = similarity - self.threshold
            confidence = min(1.0, 0.5 + margin * 2)
        else:
            # How much below threshold
            margin = self.threshold - similarity
            confidence = max(0.0, 0.5 - margin * 2)

        return verified, confidence
