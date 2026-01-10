# FaceVerifier

The main class for face verification operations.

---

## Import

```python
from faceverify import FaceVerifier
```

---

## Class Definition

```python
class FaceVerifier:
    def __init__(self, config: Optional[VerifierConfig] = None):
        ...
```

---

## Constructor

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `VerifierConfig` | `None` | Configuration object. Uses defaults if not provided. |

### Example

```python
# Default configuration
verifier = FaceVerifier()

# Custom configuration
from faceverify.config import VerifierConfig

config = VerifierConfig(
    detector_backend="retinaface",
    embedding_model="facenet",
    threshold=0.70
)
verifier = FaceVerifier(config)
```

---

## Methods

### verify()

Verify if two images contain the same person.

```python
def verify(
    self,
    image1: ImageInput,
    image2: ImageInput,
) -> VerificationResult
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `image1` | `str`, `Path`, `np.ndarray`, `PIL.Image` | First image |
| `image2` | `str`, `Path`, `np.ndarray`, `PIL.Image` | Second image |

#### Returns

`VerificationResult` - Contains verification outcome, confidence, and metadata.

#### Example

```python
result = verifier.verify("person1.jpg", "person2.jpg")

if result.verified:
    print(f"Same person! Confidence: {result.confidence:.2%}")
else:
    print(f"Different people. Similarity: {result.similarity:.4f}")
```

> [!NOTE]
> The `image` parameter accepts multiple input types: file paths (str or Path), numpy arrays, or PIL Image objects.

---

### verify_batch()

Verify multiple image pairs.

```python
def verify_batch(
    self,
    pairs: List[Tuple[ImageInput, ImageInput]],
    parallel: bool = True,
) -> List[VerificationResult]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pairs` | `List[Tuple]` | required | List of (image1, image2) tuples |
| `parallel` | `bool` | `True` | Process in parallel |

#### Example

```python
pairs = [
    ("person1_a.jpg", "person1_b.jpg"),
    ("person2_a.jpg", "person2_b.jpg"),
]
results = verifier.verify_batch(pairs)

for i, result in enumerate(results):
    print(f"Pair {i+1}: {result.verified}")
```

---

### identify()

Identify a face against a database (1:N matching).

```python
def identify(
    self,
    query_image: ImageInput,
    database: Union[str, Path, List[ImageInput]],
    top_k: int = 5,
    threshold: Optional[float] = None,
) -> IdentificationResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query_image` | `ImageInput` | required | Image to identify |
| `database` | `str`, `Path`, `List` | required | Directory or list of reference images |
| `top_k` | `int` | `5` | Number of top matches to return |
| `threshold` | `float` | `None` | Minimum similarity (uses config default) |

#### Example

```python
result = verifier.identify("unknown.jpg", "./known_faces/", top_k=3)

if result.best_match:
    print(f"Best match: {result.best_match['identity']}")
    print(f"Similarity: {result.best_match['similarity']:.2%}")
```

---

### detect_faces()

Detect faces in an image.

```python
def detect_faces(
    self,
    image: ImageInput,
    return_all: bool = False,
) -> Union[DetectionResult, List[DetectionResult]]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `ImageInput` | required | Input image |
| `return_all` | `bool` | `False` | Return all faces or just the largest |

#### Example

```python
# Get largest face
face = verifier.detect_faces("group_photo.jpg")
print(f"Face at: {face.bounding_box}")

# Get all faces
faces = verifier.detect_faces("group_photo.jpg", return_all=True)
print(f"Found {len(faces)} faces")
```

---

### extract_embedding()

Extract face embedding from an image.

```python
def extract_embedding(
    self,
    image: ImageInput,
) -> EmbeddingResult
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `ImageInput` | Image containing a face |

#### Returns

`EmbeddingResult` - Contains 512-dimensional embedding vector.

#### Example

```python
result = verifier.extract_embedding("face.jpg")
print(f"Embedding shape: {result.embedding.shape}")
print(f"Model: {result.model_name}")
```

> [!TIP]
> Embeddings can be stored and compared later without re-processing images.

---

### compare_embeddings()

Compare two face embeddings directly.

```python
def compare_embeddings(
    self,
    embedding1: np.ndarray,
    embedding2: np.ndarray,
) -> Tuple[float, float]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `embedding1` | `np.ndarray` | First embedding vector |
| `embedding2` | `np.ndarray` | Second embedding vector |

#### Returns

`Tuple[float, float]` - (similarity, distance)

#### Example

```python
emb1 = verifier.extract_embedding("face1.jpg").embedding
emb2 = verifier.extract_embedding("face2.jpg").embedding

similarity, distance = verifier.compare_embeddings(emb1, emb2)
print(f"Similarity: {similarity:.4f}")
```

---

## Properties

### threshold

Get or set the verification threshold.

```python
# Get current threshold
print(verifier.threshold)

# Set new threshold
verifier.threshold = 0.70
```

> [!CAUTION]
> Threshold must be between 0 and 1. Values outside this range raise `ValueError`.
