# ==============================================================================
# FaceVerify Docker Image
# ==============================================================================
# Production-ready container for face verification API
# ==============================================================================

FROM python:3.10-slim

# ------------------------------------------------------------------------------
# Labels
# ------------------------------------------------------------------------------
LABEL maintainer="nayandas69 <nayanchandradas@hotmail.com>"
LABEL version="1.0.0rc1"
LABEL description="FaceVerify SDK - A modular, open-source face verification library"
LABEL org.opencontainers.image.source="https://github.com/nayandas69/faceverify"

# ------------------------------------------------------------------------------
# Environment Variables
# ------------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FACEVERIFY_DETECTOR=opencv \
    FACEVERIFY_EMBEDDING_MODEL=deepface \
    FACEVERIFY_LOG_LEVEL=INFO

# ------------------------------------------------------------------------------
# Work Directory
# ------------------------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------------------------
# System Dependencies
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ------------------------------------------------------------------------------
# Python Dependencies
# ------------------------------------------------------------------------------
# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Install API server dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn>=0.22.0 \
    python-multipart>=0.0.6

# ------------------------------------------------------------------------------
# Application Code
# ------------------------------------------------------------------------------
# Copy source code
COPY . .

# Install package
RUN pip install --no-cache-dir -e .

# ------------------------------------------------------------------------------
# Create non-root user for security
# ------------------------------------------------------------------------------
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# ------------------------------------------------------------------------------
# Create directories for models and data
# ------------------------------------------------------------------------------
RUN mkdir -p /home/appuser/.faceverify/models \
    && mkdir -p /app/data

# ------------------------------------------------------------------------------
# Expose Port
# ------------------------------------------------------------------------------
EXPOSE 8000

# ------------------------------------------------------------------------------
# Health Check
# ------------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ------------------------------------------------------------------------------
# Default Command
# ------------------------------------------------------------------------------
CMD ["uvicorn", "examples.rest_api_server:app", "--host", "0.0.0.0", "--port", "8000"]
