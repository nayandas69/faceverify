"""Tests for configuration."""

import pytest
import tempfile
from pathlib import Path
from faceverify.config.settings import VerifierConfig


class TestVerifierConfig:
    """Tests for VerifierConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VerifierConfig()

        assert config.detector_backend == "mtcnn"
        assert config.embedding_model == "facenet"
        assert config.similarity_metric == "cosine"
        assert config.threshold is not None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = VerifierConfig(
            detector_backend="retinaface",
            embedding_model="arcface",
            threshold=0.70,
        )

        assert config.detector_backend == "retinaface"
        assert config.embedding_model == "arcface"
        assert config.threshold == 0.70

    def test_invalid_detector(self):
        """Test that invalid detector raises error."""
        with pytest.raises(ValueError):
            VerifierConfig(detector_backend="invalid")

    def test_invalid_embedding_model(self):
        """Test that invalid embedding model raises error."""
        with pytest.raises(ValueError):
            VerifierConfig(embedding_model="invalid")

    def test_invalid_similarity_metric(self):
        """Test that invalid similarity metric raises error."""
        with pytest.raises(ValueError):
            VerifierConfig(similarity_metric="invalid")

    def test_yaml_roundtrip(self):
        """Test saving and loading from YAML."""
        config = VerifierConfig(
            detector_backend="retinaface",
            threshold=0.70,
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config.to_yaml(f.name)
            loaded = VerifierConfig.from_yaml(f.name)

        Path(f.name).unlink()

        assert loaded.detector_backend == config.detector_backend
        assert loaded.threshold == config.threshold

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = VerifierConfig()
        d = config.to_dict()

        assert isinstance(d, dict)
        assert "detector_backend" in d
        assert "embedding_model" in d
        assert "threshold" in d
