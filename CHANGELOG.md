# Changelog

All notable changes to FaceVerify will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Additional detection backends (MTCNN, RetinaFace, MediaPipe)
- Alternative embedding models (ArcFace, VGGFace)
- Face quality assessment module
- Liveness detection integration
- Database integration for face galleries
- REST API server improvements
- Performance optimizations with GPU acceleration
- Model caching improvements

---

## [1.0.0rc1] - 2026-01-09

> **Note:** This is the first public release of FaceVerify - a Release Candidate for testing and feedback.
> Available on [PyPI](https://pypi.org/project/faceverify-sdk/) and [GitHub](https://github.com/nayandas69/faceverify/releases).

### Added

**Core Features**
- Face verification pipeline with 4-stage architecture:
  1. Face Detection - Locate faces in images
  2. Embedding Generation - Extract 512-dimensional face vectors
  3. Similarity Calculation - Compare face embeddings
  4. Decision Engine - Determine match/no-match
- Face detection using OpenCV DNN backend (default, no extra install required)
- Face embedding extraction using DeepFace with Facenet512 model
- Similarity calculation with Cosine, Euclidean, and Manhattan distance metrics
- Threshold-based decision engine with configurable threshold (default: 0.65)
- Adaptive decision maker with quality-adjusted thresholds

**CLI Interface**
- `python -m faceverify verify <img1> <img2>` - Verify two face images
- `python -m faceverify detect <img>` - Detect faces in an image
- `python -m faceverify batch <csv>` - Batch process from CSV file
- `python -m faceverify info` - Display system and library information
- `python -m faceverify --version` - Show version

**Configuration**
- YAML configuration file support (`configs/default.yaml`)
- Environment variable configuration
- Programmatic configuration via `VerifierConfig` class
- `VerificationConfig` alias for backward compatibility
- Sensible defaults that work out of the box

**Python API**
- `FaceVerifier` - Main class for face verification
- `VerificationResult` - Structured result with similarity, confidence, and metadata
- `VerifierConfig` - Configuration dataclass
- Exception classes for proper error handling

**Examples**
- `examples/basic_verification.py` - Simple usage example
- `examples/batch_processing.py` - Process multiple image pairs
- `examples/rest_api_server.py` - Flask-based REST API
- `examples/webcam_verification.py` - Real-time webcam verification

**Jupyter Notebooks**
- `notebooks/01_quickstart.ipynb` - Getting started guide
- `notebooks/02_detection_comparison.ipynb` - Compare detection backends
- `notebooks/03_embedding_analysis.ipynb` - Understand face embeddings
- `notebooks/04_threshold_tuning.ipynb` - Tune for your use case

**Documentation**
- Complete README with installation, usage, and troubleshooting
- API reference in `docs/api/`
- User guides in `docs/guides/`
- Step-by-step tutorials in `docs/tutorials/`
- Contributing guidelines

**Infrastructure**
- Docker support with multi-stage Dockerfile
- Docker Compose for easy deployment
- GitHub Actions CI/CD (test, lint, release)
- GitHub Issue templates (bug, feature, question, docs)
- PyPI publishing via Trusted Publisher (OIDC)
- Makefile for common development tasks
- Dependabot for automated dependency updates

**Testing**
- Unit tests for all modules
- Integration tests for full pipeline
- 35+ test cases with pytest
- Code coverage reporting

### Technical Details

| Component | Implementation |
|-----------|----------------|
| Detection Backend | OpenCV DNN (Caffe model) |
| Embedding Model | Facenet512 via DeepFace |
| Embedding Dimension | 512 |
| Similarity Metric | Cosine (default) |
| Default Threshold | 0.65 |
| Python Support | 3.9, 3.10, 3.11, 3.12 |

### Dependencies

| Package | Version |
|---------|---------|
| deepface | >= 0.0.79 |
| tf-keras | >= 2.15.0 |
| opencv-python | >= 4.8.0 |
| numpy | >= 1.24.0 |
| Pillow | >= 10.0.0 |
| pyyaml | >= 6.0 |

### Security
- Input validation for all public methods
- No external API calls without user consent
- Safe file path handling
- Non-root user in Docker container

### Known Limitations
- First run downloads Facenet512 model (~90MB)
- GPU acceleration not enabled by default
- Windows users may need Visual C++ Redistributable

---

## Version History

| Version | Date | Type | Description |
|---------|------|------|-------------|
| 1.0.0rc1 | 2026-01-09 | Pre-release | First public release candidate |

---

## Installation

### From PyPI (Pre-release)

```bash
pip install faceverify-sdk --pre
```

### From Source

```bash
git clone https://github.com/nayandas69/faceverify.git
cd faceverify
pip install -e .
```

### Verify Installation

```bash
python -m faceverify --version
python -m faceverify info
```

---

## Feedback

This is a release candidate. We welcome feedback and bug reports:

- Issues: https://github.com/nayandas69/faceverify/issues
- Discussions: https://github.com/nayandas69/faceverify/discussions

---

## Links

- PyPI: https://pypi.org/project/faceverify-sdk/
- Repository: https://github.com/nayandas69/faceverify
- Documentation: https://github.com/nayandas69/faceverify/tree/main/docs
- Issues: https://github.com/nayandas69/faceverify/issues
- Author: Nayan Das (https://github.com/nayandas69)
