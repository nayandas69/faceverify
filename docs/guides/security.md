# Security Best Practices

Security guidelines for FaceVerify deployments.

---

## Data Protection

### Image Storage

> [!CAUTION]
> Face images are biometric data. Handle with care.

Recommendations:

1. Encrypt stored images at rest
2. Use secure transmission (HTTPS) for APIs
3. Implement access controls
4. Consider storing only embeddings, not images

### Embedding Storage

```python
# Store embeddings instead of images
embedding = verifier.extract_embedding("face.jpg").embedding

# Save to secure storage
import numpy as np
np.save("embeddings/user_123.npy", embedding)
```

---

## API Security

### Authentication

Always require authentication for verification endpoints:

```python
from functools import wraps

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization")
        if not validate_token(auth):
            return {"error": "Unauthorized"}, 401
        return f(*args, **kwargs)
    return decorated

@app.post("/verify")
@require_auth
def verify():
    ...
```

### Rate Limiting

Prevent abuse with rate limiting:

```python
from flask_limiter import Limiter

limiter = Limiter(app, default_limits=["100 per hour"])

@app.post("/verify")
@limiter.limit("10 per minute")
def verify():
    ...
```

### Input Validation

> [!WARNING]
> Always validate uploaded images.

```python
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_image(file):
    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Invalid file type")
    
    # Check size
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        raise ValueError("File too large")
```

---

## Threshold Considerations

### False Acceptance Rate (FAR)

Higher threshold = Lower FAR = More secure

| Threshold | Approximate FAR |
|-----------|-----------------|
| 0.60 | ~1% |
| 0.70 | ~0.1% |
| 0.80 | ~0.01% |

> [!IMPORTANT]
> For security applications, use threshold >= 0.70

### Liveness Detection

FaceVerify does not include liveness detection. For production security systems:

1. Implement liveness checks separately
2. Require multiple verification factors
3. Monitor for spoofing attempts

---

## Logging

### Audit Logging

Log all verification attempts:

```python
import logging

logger = logging.getLogger("faceverify.audit")

def verify_with_audit(image1, image2, user_id):
    result = verifier.verify(image1, image2)
    
    logger.info(
        "Verification attempt",
        extra={
            "user_id": user_id,
            "verified": result.verified,
            "confidence": result.confidence,
            "timestamp": result.timestamp.isoformat(),
        }
    )
    
    return result
```

### Do Not Log

> [!CAUTION]
> Never log:
> - Raw face images
> - Full embedding vectors
> - Personal identifiable information (PII)
