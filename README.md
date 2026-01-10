<div align="center">

# FaceVerify

**A modular, open-source face verification SDK for Python.**

FaceVerify provides a complete pipeline for face detection, embedding generation, similarity comparison, and identity verification with a clean, extensible architecture.

```
Face Detection --> Embedding Generator --> Similarity Engine --> Decision Engine --> Verified / Not Verified
```

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Made in Bangladesh](https://img.shields.io/badge/Made%20in-Bangladesh-006a4e?style=flat&labelColor=f42a41)](https://en.wikipedia.org/wiki/Bangladesh)

</div>

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Models Used](#models-used)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Docker](#docker)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Multi-backend Detection** - OpenCV, MTCNN, RetinaFace, MediaPipe support
- **State-of-the-art Embeddings** - DeepFace with Facenet512 model (512-dimensional vectors)
- **Multiple Similarity Metrics** - Cosine, Euclidean, Manhattan distance calculations
- **Adaptive Decision Making** - Threshold-based and adaptive verification
- **GPU Acceleration** - Optional CUDA support for faster processing
- **Comprehensive CLI** - Command-line interface for quick verification
- **Batch Processing** - Process multiple image pairs efficiently
- **Well Tested** - Unit tests and integration tests included
- **Configurable** - YAML/JSON configuration, environment variables

---

## How It Works

FaceVerify uses a multi-stage pipeline to verify if two face images belong to the same person:

### Stage 1: Face Detection

The detector locates faces in the input images and extracts the face regions. Multiple backends are supported:

| Backend | Description | Speed | Accuracy |
|---------|-------------|-------|----------|
| OpenCV | Haar Cascade / DNN based | Fast | Good |
| MTCNN | Multi-task CNN | Medium | Very Good |
| RetinaFace | Deep learning based | Slow | Excellent |
| MediaPipe | Google MediaPipe | Fast | Good |

### Stage 2: Embedding Generation

The detected face is converted into a 512-dimensional numerical vector (embedding) that represents the unique facial features. This uses the **Facenet512** model via DeepFace.

### Stage 3: Similarity Calculation

The embeddings from both images are compared using distance metrics:

- **Cosine Similarity** - Measures angle between vectors (default)
- **Euclidean Distance** - Measures straight-line distance
- **Manhattan Distance** - Measures grid-based distance

### Stage 4: Decision Engine

The similarity score is compared against a threshold (default: 0.65) to determine if the faces match:

- Similarity >= Threshold --> **VERIFIED** (Same Person)
- Similarity < Threshold --> **NOT VERIFIED** (Different People)

---

## Models Used

| Component | Model | Details |
|-----------|-------|---------|
| Face Embedding | **Facenet512** | 512-dimensional embeddings, trained on MS-Celeb-1M and VGGFace2 |
| Face Detection | **OpenCV DNN** | Pre-trained Caffe model for face detection |
| Similarity | **Cosine Similarity** | Normalized dot product of embedding vectors |

> [!NOTE]
> The Facenet512 model weights are automatically downloaded on first use (~90MB). Subsequent runs use the cached model.

---

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Install from PyPI

```bash
pip install faceverify-sdk
```

### Install Pre-release Version

```bash
pip install faceverify-sdk --pre
```

### Install from Source

```bash
git clone https://github.com/nayandas69/faceverify.git
cd faceverify
pip install -e .
```

### Install with Development Dependencies

```bash
git clone https://github.com/nayandas69/faceverify.git
cd faceverify
pip install -e ".[dev]"
```

### Dependencies

The main dependencies are automatically installed:

```
deepface>=0.0.79
tf-keras>=2.15.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
pyyaml>=6.0
```

> [!IMPORTANT]
> On first run, DeepFace will download the Facenet512 model weights. Ensure you have an internet connection.

---

## Quick Start

### Basic Verification

```python
from faceverify import FaceVerifier

# Initialize the verifier
verifier = FaceVerifier()

# Verify two faces
result = verifier.verify("person1.jpg", "person2.jpg")

print(f"Verified: {result.verified}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Similarity: {result.similarity:.4f}")
```

### Output Example

```
Verified: True
Confidence: 92.31%
Similarity: 0.9231
```

### Advanced Configuration

```python
from faceverify import FaceVerifier
from faceverify.config import VerifierConfig

# Custom configuration
config = VerifierConfig(
    detector_backend="opencv",
    embedding_model="facenet512",
    similarity_metric="cosine",
    threshold=0.65,
    enable_gpu=False
)

verifier = FaceVerifier(config)
result = verifier.verify("image1.jpg", "image2.jpg")
```

---

## CLI Usage

FaceVerify includes a command-line interface for quick operations.

### Check Version

```bash
python -m faceverify --version
```

### Verify Two Images

```bash
python -m faceverify verify path/to/image1.jpg path/to/image2.jpg
```

### Verify with Custom Threshold

```bash
python -m faceverify verify image1.jpg image2.jpg -t 0.70
```

### Detect Faces in Image

```bash
python -m faceverify detect image.jpg
```

### Show System Info

```bash
python -m faceverify info
```

### Batch Verification from CSV

```bash
python -m faceverify batch pairs.csv -o results.json
```

CSV format:
```csv
image1,image2
path/to/face1.jpg,path/to/face2.jpg
path/to/face3.jpg,path/to/face4.jpg
```

> [!TIP]
> Use `python -m faceverify --help` to see all available commands and options.

---

## Configuration

### YAML Configuration

Create a `config.yaml` file:

```yaml
detector:
  backend: "opencv"
  confidence_threshold: 0.9
  
embedding:
  model: "facenet512"
  normalize: true
  
similarity:
  metric: "cosine"
  
decision:
  threshold: 0.65
  adaptive: false
  
performance:
  enable_gpu: false
  batch_size: 32
```

Load configuration:

```python
from faceverify import FaceVerifier
from faceverify.config import VerifierConfig

config = VerifierConfig.from_yaml("config.yaml")
verifier = FaceVerifier(config)
```

### Environment Variables

```bash
export FACEVERIFY_DETECTOR=opencv
export FACEVERIFY_EMBEDDING_MODEL=facenet512
export FACEVERIFY_THRESHOLD=0.65
export FACEVERIFY_ENABLE_GPU=false
```

---

## API Reference

### FaceVerifier Class

```python
class FaceVerifier:
    def __init__(self, config: Optional[VerifierConfig] = None)
    def verify(self, image1, image2) -> VerificationResult
    def verify_batch(self, pairs: List[Tuple]) -> List[VerificationResult]
    def identify(self, query, database) -> List[IdentificationResult]
    def extract_embedding(self, image) -> np.ndarray
    def detect_faces(self, image) -> List[Face]
```

### VerificationResult Dataclass

```python
@dataclass
class VerificationResult:
    verified: bool          # True if faces match
    confidence: float       # Confidence score (0-1)
    similarity: float       # Raw similarity score
    distance: float         # Distance between embeddings
    threshold: float        # Threshold used for decision
    detector_backend: str   # Detection backend used
    embedding_model: str    # Embedding model used
    processing_time: float  # Time taken in seconds
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detector_backend` | str | "opencv" | Face detection backend |
| `embedding_model` | str | "facenet512" | Embedding model to use |
| `similarity_metric` | str | "cosine" | Similarity calculation method |
| `threshold` | float | 0.65 | Verification threshold |
| `enable_gpu` | bool | False | Enable GPU acceleration |
| `min_face_size` | int | 20 | Minimum face size in pixels |

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage Report

```bash
pytest tests/ --cov=src/faceverify --cov-report=html
```

### Run Specific Test File

```bash
pytest tests/unit/test_similarity.py -v
```

### Run Integration Tests

```bash
pytest tests/integration/ -v
```

> [!CAUTION]
> Always run tests locally before pushing changes to ensure nothing is broken.

---

## Docker

### Build Image

```bash
docker build -t faceverify .
```

### Run Container

```bash
docker run -p 8000:8000 faceverify
```

### Docker Compose

```bash
docker-compose up
```

### Run Verification in Container

```bash
docker run -v $(pwd)/images:/images faceverify python -m faceverify verify /images/face1.jpg /images/face2.jpg
```

---

## Project Structure

```
faceverify/
├── src/
│   └── faceverify/
│       ├── __init__.py          # Package initialization
│       ├── __main__.py          # CLI entry point
│       ├── core/
│       │   ├── verifier.py      # Main FaceVerifier class
│       │   ├── pipeline.py      # Verification pipeline
│       │   └── result.py        # Result dataclasses
│       ├── detection/
│       │   ├── base.py          # Abstract detector
│       │   ├── opencv.py        # OpenCV implementation
│       │   └── factory.py       # Detector factory
│       ├── embedding/
│       │   ├── base.py          # Abstract embedder
│       │   ├── facenet.py       # DeepFace/Facenet512 implementation
│       │   └── factory.py       # Embedder factory
│       ├── similarity/
│       │   ├── metrics.py       # Distance metrics
│       │   └── engine.py        # Similarity engine
│       ├── decision/
│       │   ├── threshold.py     # Threshold-based decision
│       │   └── adaptive.py      # Adaptive threshold
│       ├── config/
│       │   ├── settings.py      # Configuration classes
│       │   └── defaults.py      # Default values
│       ├── cli/
│       │   └── __init__.py      # CLI implementation
│       ├── utils/
│       │   ├── image.py         # Image utilities
│       │   └── validators.py    # Input validation
│       └── exceptions/
│           └── errors.py        # Custom exceptions
├── tests/
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── examples/                    # Usage examples
├── configs/                     # Configuration files
├── pyproject.toml               # Project metadata
├── requirements.txt             # Dependencies
├── Dockerfile                   # Docker configuration
├── LICENSE                      # MIT License
├── CONTRIBUTING.md              # Contribution guidelines
├── CHANGELOG.md                 # Version history
└── README.md                    # This file
```

---

## Contributing

Contributions are welcome! Please follow these steps:

### Before You Start

1. Fork the repository
2. Clone your fork locally
3. Create a virtual environment and install dev dependencies

```bash
git clone https://github.com/YOUR_USERNAME/faceverify.git
cd faceverify
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Making Changes

1. Create a new branch for your feature/fix
2. Make your changes
3. Format code with Black
4. Run tests locally

```bash
git checkout -b feature/your-feature-name
# Make your changes
black src/ tests/
pytest tests/ -v
```

> [!WARNING]
> Do not push without running formatters and tests. PRs with failing tests will not be merged.

### Submitting

1. Commit with clear messages
2. Push to your fork
3. Create a Pull Request

```bash
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

### Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding or updating tests
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

> [!IMPORTANT]
> Always test your changes locally before submitting a PR. Run `pytest tests/ -v` to ensure all tests pass.

---

## Uninstallation

To remove FaceVerify from your system:

```bash
pip uninstall faceverify-sdk
```

To remove all dependencies:

```bash
pip uninstall faceverify-sdk deepface tf-keras opencv-python numpy Pillow pyyaml
```

---

## Troubleshooting

### Common Issues

**Issue: Model download fails**

> [!TIP]
> Ensure you have a stable internet connection. The Facenet512 model is downloaded from GitHub on first run.

**Issue: No face detected**

> [!NOTE]
> Ensure the image contains a clearly visible face. The minimum face size is 20x20 pixels by default.

**Issue: Low accuracy**

> [!TIP]
> Try adjusting the threshold value. Lower threshold = more strict matching. Default is 0.65.

**Issue: GPU not detected**

> [!NOTE]
> Ensure CUDA and cuDNN are properly installed. Set `enable_gpu=True` in config.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

**Nayan Das** - [@nayandas69](https://github.com/nayandas69)

Project Link: [https://github.com/nayandas69/faceverify](https://github.com/nayandas69/faceverify)

---

<p align="center">
  Made with dedication in Bangladesh by <a href="https://github.com/nayandas69">Nayan Das</a>
</p>
