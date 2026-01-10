# Configuration Guide

Complete guide to configuring FaceVerify.

---

## Configuration Methods

FaceVerify supports three configuration methods (in order of precedence):

1. Constructor arguments (highest priority)
2. Environment variables
3. YAML configuration file

---

## Quick Configuration

### Python

```python
from faceverify import FaceVerifier
from faceverify.config import VerifierConfig

config = VerifierConfig(
    detector_backend="opencv",    # Fast detection
    embedding_model="deepface",   # Uses DeepFace with Facenet512
    threshold=0.65,               # Match threshold
)

verifier = FaceVerifier(config)
```

> [!NOTE]
> `VerificationConfig` is an alias for `VerifierConfig`. Both work identically.

### Environment Variables

```bash
export FACEVERIFY_DETECTOR=opencv
export FACEVERIFY_THRESHOLD=0.65
```

### YAML File

```yaml
# config.yaml
detector:
  backend: opencv
  confidence_threshold: 0.9

embedding:
  model: deepface

decision:
  threshold: 0.65
```

```python
config = VerifierConfig.from_yaml("config.yaml")
```

---

## Detector Backends

### OpenCV (Default)

- Fastest option
- No extra installation
- Good for real-time applications

```python
config = VerifierConfig(detector_backend="opencv")
```

### MTCNN

- Better accuracy
- Requires: `pip install mtcnn`

```python
config = VerifierConfig(detector_backend="mtcnn")
```

### RetinaFace

- Best accuracy
- Slower than others
- Requires: `pip install retinaface`

```python
config = VerifierConfig(detector_backend="retinaface")
```

### MediaPipe

- Good balance of speed/accuracy
- Requires: `pip install mediapipe`

```python
config = VerifierConfig(detector_backend="mediapipe")
```

> [!TIP]
> Use `opencv` for speed, `retinaface` for accuracy.

---

## Embedding Models

### DeepFace (Default)

- Uses Facenet512 model internally
- 512-dimensional embeddings
- Best overall performance

```python
config = VerifierConfig(embedding_model="deepface")
```

> [!IMPORTANT]
> The embedding model uses DeepFace library with Facenet512. First run downloads model weights (~90MB).

### Facenet

- Alias for deepface with Facenet512
- 512-dimensional embeddings

```python
config = VerifierConfig(embedding_model="facenet")
```

---

## Similarity Metrics

### Cosine (Default)

- Range: 0 to 1 (higher = more similar)
- Best for normalized embeddings

```python
config = VerifierConfig(similarity_metric="cosine")
```

### Euclidean

- Range: 0 to infinity (lower = more similar)
- Sensitive to magnitude

```python
config = VerifierConfig(similarity_metric="euclidean")
```

### Manhattan

- Range: 0 to infinity (lower = more similar)
- Less sensitive to outliers

```python
config = VerifierConfig(similarity_metric="manhattan")
```

---

## Threshold Tuning

### Understanding Thresholds

| Threshold | Effect |
|-----------|--------|
| Higher (0.7+) | Stricter matching, fewer false positives |
| Lower (0.5-) | Looser matching, fewer false negatives |

### Use Cases

| Use Case | Recommended Threshold |
|----------|----------------------|
| High security | 0.75 - 0.80 |
| Standard verification | 0.60 - 0.70 |
| Loose matching | 0.50 - 0.55 |

> [!IMPORTANT]
> Always test with your specific dataset to find optimal threshold.

### Setting Threshold

```python
# At initialization
config = VerifierConfig(threshold=0.70)

# Or modify later
verifier.threshold = 0.70
```

---

## Performance Settings

### GPU Acceleration

```python
config = VerifierConfig(enable_gpu=True)
```

> [!NOTE]
> GPU requires CUDA-compatible TensorFlow installation.

### Batch Size

```python
config = VerifierConfig(batch_size=32)
```

Larger batch sizes improve throughput but use more memory.

---

## Complete Example

```python
from faceverify import FaceVerifier
from faceverify.config import VerifierConfig

# High-accuracy configuration
config = VerifierConfig(
    # Detection
    detector_backend="retinaface",
    detector_confidence=0.95,
    
    # Embedding
    embedding_model="deepface",
    normalize_embeddings=True,
    
    # Similarity
    similarity_metric="cosine",
    
    # Decision
    threshold=0.70,
    
    # Performance
    enable_gpu=True,
    batch_size=16,
)

verifier = FaceVerifier(config)
