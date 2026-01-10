"""Integration tests for face verification."""

import pytest
import numpy as np
from faceverify import FaceVerifier
from faceverify.config import VerifierConfig


@pytest.mark.integration
class TestFaceVerifier:
    """Integration tests for FaceVerifier."""

    def test_initialization(self):
        """Test verifier initialization."""
        verifier = FaceVerifier()

        assert verifier.config is not None
        assert verifier.threshold > 0

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = VerifierConfig(
            detector_backend="opencv",
            threshold=0.70,
        )

        verifier = FaceVerifier(config)

        assert verifier.config.detector_backend == "opencv"
        assert verifier.threshold == 0.70

    def test_threshold_setter(self):
        """Test updating threshold."""
        verifier = FaceVerifier()
        verifier.threshold = 0.80

        assert verifier.threshold == 0.80

    def test_threshold_validation(self):
        """Test threshold validation."""
        verifier = FaceVerifier()

        with pytest.raises(ValueError):
            verifier.threshold = 1.5
