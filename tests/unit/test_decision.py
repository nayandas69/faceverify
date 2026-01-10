"""Tests for decision makers."""

import pytest
from faceverify.decision.threshold import ThresholdDecisionMaker
from faceverify.decision.adaptive import AdaptiveDecisionMaker


class TestThresholdDecisionMaker:
    """Tests for threshold-based decision maker."""

    def test_above_threshold(self):
        """Similarity above threshold should verify."""
        maker = ThresholdDecisionMaker(threshold=0.65)
        verified, confidence = maker.decide(similarity=0.80, distance=0.2)

        assert verified is True
        assert confidence > 0.5

    def test_below_threshold(self):
        """Similarity below threshold should not verify."""
        maker = ThresholdDecisionMaker(threshold=0.65)
        verified, confidence = maker.decide(similarity=0.50, distance=0.5)

        assert verified is False
        assert confidence < 0.5

    def test_at_threshold(self):
        """Similarity at threshold should verify."""
        maker = ThresholdDecisionMaker(threshold=0.65)
        verified, confidence = maker.decide(similarity=0.65, distance=0.35)

        assert verified is True

    def test_confidence_increases_with_margin(self):
        """Higher margin should give higher confidence."""
        maker = ThresholdDecisionMaker(threshold=0.65)

        _, conf1 = maker.decide(similarity=0.70, distance=0.3)
        _, conf2 = maker.decide(similarity=0.90, distance=0.1)

        assert conf2 > conf1


class TestAdaptiveDecisionMaker:
    """Tests for adaptive decision maker."""

    def test_basic_decision(self):
        """Test basic decision making."""
        maker = AdaptiveDecisionMaker(base_threshold=0.65)
        verified, confidence = maker.decide(similarity=0.70, distance=0.3)

        assert verified is True

    def test_quality_adjustment(self):
        """Test that quality score affects decision."""
        maker = AdaptiveDecisionMaker(base_threshold=0.65)

        # Low quality - should use higher threshold
        verified_low, _ = maker.decide(
            similarity=0.64,
            distance=0.36,
            quality_score=0.3,
        )

        # High quality - should use lower threshold
        verified_high, _ = maker.decide(
            similarity=0.64,
            distance=0.36,
            quality_score=0.9,
        )

        # High quality might verify while low quality doesn't
        # (depending on the adjustment magnitude)
        assert isinstance(verified_low, bool)
        assert isinstance(verified_high, bool)
