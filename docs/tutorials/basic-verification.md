# Basic Verification Tutorial

Learn to verify faces with FaceVerify.

---

## Prerequisites

- FaceVerify installed
- Two or more face images

---

## Step 1: Setup

```python
from faceverify import FaceVerifier

# Initialize with defaults
verifier = FaceVerifier()
```

---

## Step 2: Verify Two Images

```python
result = verifier.verify("person1_photo_a.jpg", "person1_photo_b.jpg")

print(f"Verified: {result.verified}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Similarity: {result.similarity:.4f}")
```

---

## Step 3: Interpret Results

### Same Person (Match)

```
Verified: True
Confidence: 92.31%
Similarity: 0.9231
```

The similarity score (0.9231) exceeds the threshold (0.65), so `verified` is `True`.

### Different People (No Match)

```
Verified: False
Confidence: 51.08%
Similarity: 0.5108
```

The similarity score (0.5108) is below the threshold (0.65), so `verified` is `False`.

---

## Step 4: Custom Threshold

Adjust sensitivity:

```python
from faceverify.config import VerifierConfig

# Stricter matching
config = VerifierConfig(threshold=0.75)
verifier = FaceVerifier(config)

result = verifier.verify("image1.jpg", "image2.jpg")
```

---

## Step 5: Handle Errors

```python
try:
    result = verifier.verify("image1.jpg", "image2.jpg")
    print(f"Result: {result.verified}")
except FileNotFoundError as e:
    print(f"Image not found: {e}")
except ValueError as e:
    print(f"Invalid image: {e}")
except Exception as e:
    print(f"Verification failed: {e}")
```

---

## Complete Example

```python
"""Basic face verification example."""

from faceverify import FaceVerifier
from faceverify.config import VerifierConfig
import sys


def main():
    # Parse arguments
    if len(sys.argv) != 3:
        print("Usage: python verify.py <image1> <image2>")
        sys.exit(1)
    
    image1, image2 = sys.argv[1], sys.argv[2]
    
    # Initialize
    config = VerifierConfig(
        detector_backend="opencv",
        threshold=0.65,
    )
    verifier = FaceVerifier(config)
    
    # Verify
    try:
        result = verifier.verify(image1, image2)
        
        print("\n" + "=" * 50)
        print("  Verification Result")
        print("=" * 50)
        
        status = "MATCH" if result.verified else "NO MATCH"
        print(f"  Status:     {status}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Similarity: {result.similarity:.4f}")
        print(f"  Threshold:  {result.threshold:.4f}")
        print(f"  Time:       {result.processing_time:.3f}s")
        
        print("=" * 50 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## Next Steps

- [Batch Processing](batch-processing.md) - Process multiple pairs
- [Configuration Guide](../guides/configuration.md) - Tune settings
