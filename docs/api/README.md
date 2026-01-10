# API Reference

Complete API documentation for FaceVerify.

---

## Contents

| Module | Description |
|--------|-------------|
| [FaceVerifier](faceverifier.md) | Main verification class |
| [Configuration](configuration.md) | Configuration classes |
| [Results](results.md) | Result dataclasses |
| [CLI](cli.md) | Command line interface |

---

## Quick Import Reference

```python
# Main class
from faceverify import FaceVerifier

# Configuration
from faceverify import VerifierConfig
from faceverify.config import VerifierConfig, VerificationConfig  # Alias

# Result types
from faceverify import (
    VerificationResult,
    DetectionResult,
    EmbeddingResult,
    IdentificationResult,
)

# Exceptions
from faceverify import (
    FaceVerifyError,
    NoFaceDetectedError,
    MultipleFacesError,
    InvalidImageError,
    ModelNotFoundError,
)

# Detection factory
from faceverify.detection import create_detector

# Similarity metrics
from faceverify.similarity.metrics import (
    cosine_similarity,
    euclidean_distance,
    manhattan_distance,
)
```

---

## Module Hierarchy

```
faceverify/
|-- __init__.py           # Main exports
|-- __main__.py           # CLI entry (python -m faceverify)
|
|-- core/
|   |-- verifier.py       # FaceVerifier class
|   |-- pipeline.py       # VerificationPipeline
|   |-- result.py         # Result dataclasses
|
|-- config/
|   |-- settings.py       # VerifierConfig class
|   |-- defaults.py       # Default constants
|
|-- detection/
|   |-- base.py           # BaseDetector abstract class
|   |-- factory.py        # create_detector() function
|   |-- opencv.py         # OpenCVDetector
|   |-- mtcnn.py          # MTCNNDetector
|   |-- retinaface.py     # RetinaFaceDetector
|   |-- mediapipe.py      # MediaPipeDetector
|
|-- embedding/
|   |-- base.py           # BaseEmbedding abstract class
|   |-- factory.py        # create_embedder() function
|   |-- facenet.py        # FacenetEmbedding (DeepFace)
|   |-- arcface.py        # ArcFaceEmbedding
|   |-- vggface.py        # VGGFaceEmbedding
|
|-- similarity/
|   |-- metrics.py        # cosine, euclidean, manhattan
|   |-- engine.py         # SimilarityEngine
|
|-- decision/
|   |-- base.py           # BaseDecisionMaker
|   |-- threshold.py      # ThresholdDecisionMaker
|   |-- adaptive.py       # AdaptiveDecisionMaker
|
|-- preprocessing/
|   |-- alignment.py      # Face alignment
|   |-- normalization.py  # Image normalization
|
|-- exceptions/
|   |-- errors.py         # Custom exceptions
|
|-- utils/
|   |-- image.py          # Image loading utilities
|   |-- model_loader.py   # Model download/cache
|   |-- validators.py     # Input validation
|   |-- gpu.py            # GPU detection
|
|-- cli/
    |-- __init__.py       # CLI implementation
```

---

## Exports from Main Package

The following are exported from `faceverify`:

| Export | Type | Description |
|--------|------|-------------|
| `FaceVerifier` | class | Main verification class |
| `VerifierConfig` | class | Configuration class |
| `VerificationResult` | dataclass | Verification result |
| `DetectionResult` | dataclass | Detection result |
| `EmbeddingResult` | dataclass | Embedding result |
| `IdentificationResult` | dataclass | 1:N identification result |
| `FaceVerifyError` | exception | Base exception |
| `NoFaceDetectedError` | exception | No face found |
| `MultipleFacesError` | exception | Multiple faces found |
| `InvalidImageError` | exception | Invalid image input |
| `ModelNotFoundError` | exception | Model not found |
| `__version__` | str | Package version |
| `__author__` | str | Package author |
