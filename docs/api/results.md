# Result Classes

Dataclasses for representing verification results.

---

## VerificationResult

Result of face verification between two images.

### Import

```python
from faceverify import VerificationResult
# or
from faceverify.core.result import VerificationResult
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `verified` | `bool` | Whether faces match (similarity >= threshold) |
| `confidence` | `float` | Confidence score (0-1) |
| `similarity` | `float` | Similarity score (0-1) |
| `distance` | `float` | Distance between embeddings |
| `threshold` | `float` | Threshold used for decision |
| `detector_backend` | `str` | Detection backend used |
| `embedding_model` | `str` | Embedding model used |
| `similarity_metric` | `str` | Similarity metric used |
| `processing_time` | `float` | Time taken in seconds |
| `timestamp` | `datetime` | When verification occurred |
| `metadata` | `Dict[str, Any]` | Additional metadata |

### Methods

#### to_dict()

Convert result to dictionary (JSON-serializable).

```python
def to_dict(self) -> Dict[str, Any]
```

### Example

```python
result = verifier.verify("person1.jpg", "person2.jpg")

print(f"Verified: {result.verified}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Similarity: {result.similarity:.4f}")
print(f"Distance: {result.distance:.4f}")
print(f"Threshold: {result.threshold}")
print(f"Time: {result.processing_time:.3f}s")

# Convert to dict for JSON
import json
print(json.dumps(result.to_dict(), indent=2))
```

> [!NOTE]
> The `similarity` attribute is what you compare against the threshold. Higher similarity means more likely to be the same person.

---

## DetectionResult

Result of face detection.

### Import

```python
from faceverify import DetectionResult
# or
from faceverify.core.result import DetectionResult
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `bounding_box` | `BoundingBox` | Face location in image |
| `confidence` | `float` | Detection confidence (0-1) |
| `landmarks` | `Landmarks` | Facial landmarks (optional) |
| `face_image` | `np.ndarray` | Cropped face image (optional) |
| `aligned_face` | `np.ndarray` | Aligned face image (optional) |
| `metadata` | `Dict[str, Any]` | Additional metadata |

### Example

```python
detection = verifier.detect_faces("photo.jpg")

print(f"Face at: {detection.bounding_box.to_tuple()}")
print(f"Confidence: {detection.confidence:.2%}")

if detection.landmarks:
    print(f"Left eye: {detection.landmarks.left_eye}")
    print(f"Right eye: {detection.landmarks.right_eye}")
```

---

## BoundingBox

Represents a face bounding box.

### Import

```python
from faceverify.core.result import BoundingBox
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `x` | `int` | Top-left X coordinate |
| `y` | `int` | Top-left Y coordinate |
| `width` | `int` | Box width in pixels |
| `height` | `int` | Box height in pixels |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `x1` | `int` | Left edge (same as x) |
| `y1` | `int` | Top edge (same as y) |
| `x2` | `int` | Right edge (x + width) |
| `y2` | `int` | Bottom edge (y + height) |
| `center` | `Tuple[int, int]` | Center point (cx, cy) |
| `area` | `int` | Area in pixels |

### Methods

```python
bbox.to_tuple()  # Returns (x, y, width, height)
bbox.to_xyxy()   # Returns (x1, y1, x2, y2)
```

### Example

```python
bbox = detection.bounding_box

# Get coordinates
x, y, w, h = bbox.to_tuple()
x1, y1, x2, y2 = bbox.to_xyxy()

# Crop face from image
import cv2
image = cv2.imread("photo.jpg")
face_crop = image[y1:y2, x1:x2]
```

---

## Landmarks

Facial landmarks (5-point).

### Import

```python
from faceverify.core.result import Landmarks
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `left_eye` | `Tuple[float, float]` | Left eye center (x, y) |
| `right_eye` | `Tuple[float, float]` | Right eye center (x, y) |
| `nose` | `Tuple[float, float]` | Nose tip (x, y) |
| `left_mouth` | `Tuple[float, float]` | Left mouth corner (x, y) |
| `right_mouth` | `Tuple[float, float]` | Right mouth corner (x, y) |

### Methods

```python
landmarks.to_dict()  # Returns dict of all landmarks
```

> [!NOTE]
> Not all detectors provide landmarks. Check if `landmarks is not None` before accessing.

---

## EmbeddingResult

Result of face embedding extraction.

### Import

```python
from faceverify import EmbeddingResult
# or
from faceverify.core.result import EmbeddingResult
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `embedding` | `np.ndarray` | Face embedding vector |
| `model_name` | `str` | Model used for extraction |
| `dimension` | `int` | Embedding dimension (e.g., 512) |
| `normalized` | `bool` | Whether L2-normalized |
| `processing_time` | `float` | Extraction time in seconds |
| `metadata` | `Dict[str, Any]` | Additional metadata |

### Example

```python
result = verifier.extract_embedding("face.jpg")

print(f"Shape: {result.embedding.shape}")      # (512,)
print(f"Model: {result.model_name}")           # facenet
print(f"Dimension: {result.dimension}")        # 512
print(f"Normalized: {result.normalized}")      # True
print(f"Time: {result.processing_time:.3f}s")

# Store embedding for later comparison
import numpy as np
np.save("embedding.npy", result.embedding)
```

> [!TIP]
> Embeddings can be stored in a database and compared later without re-processing images. This is useful for building face databases.

---

## IdentificationResult

Result of 1:N face identification.

### Import

```python
from faceverify import IdentificationResult
# or
from faceverify.core.result import IdentificationResult
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `query_image` | `str` | Query image path |
| `matches` | `List[Dict]` | List of matches (sorted by similarity) |
| `best_match` | `Dict` | Top match or `None` |
| `total_candidates` | `int` | Database size searched |
| `processing_time` | `float` | Total time in seconds |
| `timestamp` | `datetime` | When identification occurred |

### Match Dictionary Structure

Each match in `matches` contains:

```python
{
    "identity": str,      # Image path or identifier
    "similarity": float,  # Similarity score (0-1)
    "distance": float,    # Distance score
}
```

### Methods

```python
result.to_dict()  # Returns JSON-serializable dict
```

### Example

```python
result = verifier.identify("unknown.jpg", "./known_faces/", top_k=3)

print(f"Searched {result.total_candidates} candidates")
print(f"Found {len(result.matches)} matches")

if result.best_match:
    print(f"Best match: {result.best_match['identity']}")
    print(f"Similarity: {result.best_match['similarity']:.2%}")
else:
    print("No match found above threshold")

# Iterate all matches
for match in result.matches:
    print(f"  {match['identity']}: {match['similarity']:.2%}")
