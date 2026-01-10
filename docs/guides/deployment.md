# Deployment Guide

Deploy FaceVerify to production.

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run
CMD ["python", "-m", "faceverify.api"]
```

### Build and Run

```bash
docker build -t faceverify:latest .
docker run -p 8000:8000 faceverify:latest
```

---

## Environment Configuration

### Production Settings

```bash
# .env
FACEVERIFY_DETECTOR=retinaface
FACEVERIFY_THRESHOLD=0.70
FACEVERIFY_ENABLE_GPU=true
FACEVERIFY_BATCH_SIZE=16
```

---

## Scaling

### Horizontal Scaling

FaceVerify is stateless - scale horizontally with load balancer:

```
Load Balancer
    |
    +-- FaceVerify Instance 1
    +-- FaceVerify Instance 2
    +-- FaceVerify Instance 3
```

### GPU Scaling

For high throughput:

1. Use GPU instances
2. Enable batching
3. Consider model serving (TensorFlow Serving)

---

## Monitoring

### Health Check Endpoint

```python
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": __version__,
    }
```

### Metrics to Monitor

| Metric | Description |
|--------|-------------|
| Verification latency | p50, p95, p99 response times |
| Error rate | Failed verifications / total |
| Throughput | Verifications per second |
| Memory usage | RAM consumption |
| GPU utilization | If using GPU |

---

## Troubleshooting

### High Memory Usage

- Reduce batch size
- Process images sequentially
- Use smaller embedding model

### Slow Performance

- Enable GPU
- Use faster detector (opencv)
- Pre-compute embeddings

### Model Loading Errors

> [!TIP]
> First run downloads model weights (~90MB). Ensure:
> - Internet connectivity
> - Write access to cache directory
> - Sufficient disk space
