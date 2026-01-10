# FaceVerify Documentation

A modular, open-source face verification SDK for Python.

---

## Overview

FaceVerify is an open-source Python library for face verification, providing a complete pipeline from face detection to identity verification. Built with accuracy and ease-of-use in mind.

---

## Quick Links

| Section | Description |
|---------|-------------|
| [Getting Started](guides/getting-started.md) | Installation and first steps |
| [API Reference](api/README.md) | Complete API documentation |
| [Tutorials](tutorials/README.md) | Step-by-step tutorials |
| [Configuration](guides/configuration.md) | Configuration options |

---

## Features

- Face detection with multiple backends (OpenCV, MTCNN, RetinaFace, MediaPipe)
- Face embedding extraction using deep learning models (Facenet512, ArcFace, VGG-Face)
- Multiple similarity metrics (Cosine, Euclidean, Manhattan)
- Configurable decision thresholds
- Batch processing support
- CLI and REST API interfaces
- Comprehensive error handling

---

## Installation

```bash
pip install faceverify-sdk
```

For pre-release:

```bash
pip install faceverify-sdk --pre
```

For development:

```bash
git clone https://github.com/nayandas69/faceverify.git
cd faceverify
pip install -e ".[dev]"
```

---

## Basic Usage

```python
from faceverify import FaceVerifier

verifier = FaceVerifier()
result = verifier.verify("person1.jpg", "person2.jpg")

print(f"Verified: {result.verified}")
print(f"Confidence: {result.confidence:.2%}")
```

---

## Architecture

```
Input Images
     |
     v
+------------------+
|  Face Detection  |  --> Locate faces in images
+------------------+
     |
     v
+------------------+
| Embedding Model  |  --> Extract 512-dim feature vectors
+------------------+
     |
     v
+------------------+
| Similarity Calc  |  --> Compare embeddings (cosine similarity)
+------------------+
     |
     v
+------------------+
| Decision Engine  |  --> Apply threshold, return result
+------------------+
     |
     v
  VERIFIED / NOT VERIFIED
```

---

## Support

- GitHub Issues: https://github.com/nayandas69/faceverify/issues
- Email: nayanchandradas@hotmail.com

---

## License

MIT License - See [LICENSE](../LICENSE) for details.
