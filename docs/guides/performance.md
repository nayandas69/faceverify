# Performance Tuning

Optimize FaceVerify for your use case.

---

## Benchmarks

Approximate performance on CPU (Intel i7):

| Backend | Detection Time | Notes |
|---------|---------------|-------|
| OpenCV | ~20ms | Fastest |
| MediaPipe | ~50ms | Good balance |
| MTCNN | ~100ms | Accurate |
| RetinaFace | ~200ms | Most accurate |

| Model | Embedding Time | Notes |
|-------|---------------|-------|
| Facenet | ~150ms | Default |
| ArcFace | ~180ms | Similar |
| VGGFace | ~250ms | Larger model |

---

## Optimization Strategies

### 1. Choose Appropriate Backend

For real-time applications:

```python
config = VerifierConfig(detector_backend="opencv")
```

For batch processing:

```python
config = VerifierConfig(detector_backend="retinaface")
```

### 2. Enable GPU

```python
config = VerifierConfig(enable_gpu=True)
```

> [!IMPORTANT]
> Requires CUDA-compatible GPU and TensorFlow with GPU support.

### 3. Batch Processing

Process multiple pairs at once:

```python
pairs = [("a1.jpg", "a2.jpg"), ("b1.jpg", "b2.jpg"), ...]
results = verifier.verify_batch(pairs, parallel=True)
```

### 4. Cache Embeddings

Extract and store embeddings for repeated comparisons:

```python
# Extract once
embeddings = {}
for image_path in images:
    result = verifier.extract_embedding(image_path)
    embeddings[image_path] = result.embedding

# Compare many times
similarity, _ = verifier.compare_embeddings(
    embeddings["person1.jpg"],
    embeddings["person2.jpg"]
)
```

### 5. Reduce Image Size

Resize large images before processing:

```python
import cv2

image = cv2.imread("large_image.jpg")
resized = cv2.resize(image, (640, 480))
result = verifier.verify(resized, "other.jpg")
```

---

## Memory Optimization

### Limit Batch Size

```python
config = VerifierConfig(batch_size=8)  # Reduce if memory limited
```

### Process Sequentially

```python
results = verifier.verify_batch(pairs, parallel=False)
```

---

## Profiling

Measure processing time:

```python
import time

start = time.time()
result = verifier.verify("a.jpg", "b.jpg")
elapsed = time.time() - start

print(f"Verification took: {elapsed:.3f}s")
print(f"Internal time: {result.processing_time:.3f}s")
