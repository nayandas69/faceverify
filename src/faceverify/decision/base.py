"""Abstract base class for decision makers."""

from abc import ABC, abstractmethod
from typing import Tuple


class BaseDecisionMaker(ABC):
    """
    Abstract base class for verification decision making.

    Decision makers take similarity scores and determine whether
    two faces belong to the same person.
    """

    @abstractmethod
    def decide(
        self,
        similarity: float,
        distance: float,
    ) -> Tuple[bool, float]:
        """
        Make verification decision.

        Args:
            similarity: Similarity score (0 to 1)
            distance: Distance between embeddings

        Returns:
            Tuple of (verified, confidence)
        """
        pass
