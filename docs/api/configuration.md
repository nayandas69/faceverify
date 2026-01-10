# Configuration

Configuration classes for FaceVerify.

---

## VerifierConfig

Main configuration class for FaceVerifier.

### Import

```python
from faceverify import VerifierConfig
# or
from faceverify.config import VerifierConfig
```

> [!NOTE]
> `VerificationConfig` is an alias for `VerifierConfig` for backward compatibility.

### Constructor

```python
VerifierConfig(
    detector_backend: str = "mtcnn",
    detector_confidence: float = 0.9,
    embedding_model: str = "facenet",
    normalize_embeddings: bool = True,
    similarity_metric: str = "cosine",
    threshold: Optional[float] = None,
    face_size: Tuple[int, int] = (160, 160),
    enable_gpu: bool = True,
    batch_size: int = 32,
    metadata: Dict[str, Any] = {},
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detector_backend` | `str` | `"mtcnn"` | Face detection backend |
| `detector_confidence` | `float` | `0.9` | Minimum detection confidence (0-1) |
| `embedding_model` | `str` | `"facenet"` | Embedding model name |
| `normalize_embeddings` | `bool` | `True` | L2-normalize embeddings |
| `similarity_metric` | `str` | `"cosine"` | Similarity metric |
| `threshold` | `float` | `None` | Verification threshold (auto-set if None) |
| `face_size` | `Tuple[int, int]` | `(160, 160)` | Target face size for preprocessing |
| `enable_gpu` | `bool` | `True` | Enable GPU acceleration |
| `batch_size` | `int` | `32` | Batch processing size |
| `metadata` | `Dict` | `{}` | Extra metadata storage |

### Example

```python
from faceverify import FaceVerifier, VerifierConfig

# Default configuration
config = VerifierConfig()

# Custom configuration
config = VerifierConfig(
    detector_backend="opencv",      # Fastest, no extra install
    detector_confidence=0.8,
    embedding_model="facenet",      # Uses DeepFace Facenet512
    similarity_metric="cosine",
    threshold=0.65,
    enable_gpu=True,
)

verifier = FaceVerifier(config)
```

---

## Supported Values

### Detector Backends

| Backend | Description | Requirements |
|---------|-------------|--------------|
| `opencv` | OpenCV Haar cascades | Built-in (no extra install) |
| `mtcnn` | Multi-task CNN | `pip install mtcnn` |
| `retinaface` | RetinaFace detector | `pip install retinaface` |
| `mediapipe` | MediaPipe face detection | `pip install mediapipe` |

> [!TIP]
> Use `opencv` for fastest detection with no extra dependencies. Use `retinaface` for best accuracy.

### Embedding Models

| Model | Dimensions | Description |
|-------|------------|-------------|
| `facenet` | 512 | Facenet512 via DeepFace library |
| `arcface` | 512 | ArcFace model |
| `vggface` | 2048 | VGG-Face model |

> [!IMPORTANT]
> The `facenet` model uses the DeepFace library internally with the Facenet512 architecture for accurate 512-dimensional embeddings.

### Similarity Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| `cosine` | 0-1 | Cosine similarity (recommended) |
| `euclidean` | 0-inf | Euclidean (L2) distance |
| `manhattan` | 0-inf | Manhattan (L1) distance |

---

## Default Thresholds

Optimized thresholds for model/metric combinations:

| Model | Cosine | Euclidean |
|-------|--------|-----------|
| facenet | 0.65 | 0.55 |
| arcface | 0.68 | 0.52 |
| vggface | 0.60 | 0.58 |

> [!NOTE]
> If `threshold` is `None` in the constructor, the optimal default is automatically selected based on the `embedding_model` and `similarity_metric`.

---

## Methods

### from_yaml()

Load configuration from a YAML file.

```python
@classmethod
def from_yaml(cls, path: str | Path) -> VerifierConfig
```

#### Example

```yaml
# config.yaml
detector:
  backend: opencv
  confidence_threshold: 0.9

embedding:
  model: facenet
  normalize: true

similarity:
  metric: cosine

decision:
  threshold: 0.65

performance:
  enable_gpu: true
  batch_size: 32
```

```python
config = VerifierConfig.from_yaml("config.yaml")
verifier = FaceVerifier(config)
```

### to_yaml()

Save configuration to a YAML file.

```python
def to_yaml(self, path: str | Path) -> None
```

#### Example

```python
config = VerifierConfig(threshold=0.70)
config.to_yaml("my_config.yaml")
```

### to_dict()

Convert configuration to dictionary.

```python
def to_dict(self) -> Dict[str, Any]
```

---

## Environment Variables

Configuration can be set via environment variables (prefix: `FACEVERIFY_`):

| Variable | Config Key | Type |
|----------|------------|------|
| `FACEVERIFY_DETECTOR` | `detector_backend` | str |
| `FACEVERIFY_DETECTOR_CONFIDENCE` | `detector_confidence` | float |
| `FACEVERIFY_EMBEDDING_MODEL` | `embedding_model` | str |
| `FACEVERIFY_NORMALIZE_EMBEDDINGS` | `normalize_embeddings` | bool |
| `FACEVERIFY_SIMILARITY_METRIC` | `similarity_metric` | str |
| `FACEVERIFY_THRESHOLD` | `threshold` | float |
| `FACEVERIFY_ENABLE_GPU` | `enable_gpu` | bool |
| `FACEVERIFY_BATCH_SIZE` | `batch_size` | int |

> [!NOTE]
> Priority order: Constructor arguments > Environment variables > File configuration > Defaults

---

## Validation

The configuration validates:

- `detector_backend` must be in `["mtcnn", "retinaface", "mediapipe", "opencv"]`
- `embedding_model` must be in `["facenet", "arcface", "vggface"]`
- `similarity_metric` must be in `["cosine", "euclidean", "manhattan"]`
- `detector_confidence` must be between 0 and 1
- `threshold` must be between 0 and 1 (if provided)

Invalid values raise `ValueError`.
