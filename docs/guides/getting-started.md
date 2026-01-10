# Getting Started

Get up and running with FaceVerify in minutes.

---

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)

---

## Installation

### From PyPI

```bash
pip install faceverify-sdk
```

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

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

---

## Verify Installation

```bash
python -m faceverify info
```

Expected output:

```
FaceVerify v1.0.0rc1
========================================
Python:        3.x.x
Platform:      Windows/Linux/macOS
...
```

---

## First Verification

### Step 1: Prepare Images

Create a folder with test images:

```
test_images/
  person1_a.jpg   # First photo of person 1
  person1_b.jpg   # Second photo of person 1
  person2.jpg     # Photo of different person
```

> [!TIP]
> Use clear, front-facing photos with good lighting for best results.

### Step 2: Run Verification

#### Using Python

```python
from faceverify import FaceVerifier

# Initialize
verifier = FaceVerifier()

# Verify same person
result = verifier.verify("test_images/person1_a.jpg", "test_images/person1_b.jpg")
print(f"Same person: {result.verified}")
print(f"Confidence: {result.confidence:.2%}")

# Verify different people
result = verifier.verify("test_images/person1_a.jpg", "test_images/person2.jpg")
print(f"Same person: {result.verified}")
print(f"Confidence: {result.confidence:.2%}")
```

#### Using CLI

```bash
# Same person
python -m faceverify verify test_images/person1_a.jpg test_images/person1_b.jpg

# Different people
python -m faceverify verify test_images/person1_a.jpg test_images/person2.jpg
```

---

## Understanding Results

### VerificationResult Fields

| Field | Description |
|-------|-------------|
| `verified` | True if faces match |
| `confidence` | Confidence score (0-1) |
| `similarity` | Raw similarity score |
| `threshold` | Threshold used |

### Example Output

```
==================================================
  Face Verification Result
==================================================
  Status:      VERIFIED
  Confidence:  92.31%
  Similarity:  0.9231
  Threshold:   0.6500
==================================================
```

---

## Next Steps

1. [Configure detection and embedding backends](configuration.md)
2. [Learn about batch processing](../tutorials/batch-processing.md)
3. [Build a REST API](../tutorials/rest-api.md)
4. [Optimize performance](performance.md)

---

## Troubleshooting

### No faces detected

> [!WARNING]
> If verification fails with "No face detected", check:
> - Image has a clear, visible face
> - Image is not too dark or blurry
> - Face is not too small (at least 80x80 pixels)

### Incorrect results

> [!TIP]
> If verification gives unexpected results:
> - Try different detector backends: `-d retinaface`
> - Adjust threshold: `-t 0.70`
> - Use higher quality images

### Import errors

> [!CAUTION]
> If you see `ModuleNotFoundError`:
> - Ensure faceverify-sdk is installed: `pip install faceverify-sdk`
> - Check Python version: `python --version`
> - Activate virtual environment if using one
