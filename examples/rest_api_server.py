#!/usr/bin/env python3
"""
REST API Server Example
========================

This example demonstrates how to create a REST API
for face verification using FastAPI.

Installation:
    pip install fastapi uvicorn python-multipart

Usage:
    uvicorn rest_api_server:app --reload --port 8000

Endpoints:
    POST /verify         - Verify two faces (file upload)
    POST /verify/base64  - Verify two faces (base64 encoded)
    POST /detect         - Detect faces in image
    POST /embedding      - Extract face embedding
    GET  /health         - Health check
    GET  /info           - API information
"""

import sys
from typing import Optional
from io import BytesIO
import base64

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Form
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    print("Error: FastAPI not installed.")
    print("Install with: pip install fastapi uvicorn python-multipart")
    sys.exit(1)

import numpy as np
from PIL import Image

from faceverify import FaceVerifier, __version__
from faceverify.config import VerifierConfig


# Initialize FastAPI app
app = FastAPI(
    title="FaceVerify API",
    description="REST API for face verification",
    version=__version__,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize verifier
config = VerifierConfig(
    detector_backend="opencv",
    embedding_model="facenet",
)
verifier = FaceVerifier(config)


# Response models
class VerifyResponse(BaseModel):
    """Response model for verification endpoint."""
    verified: bool
    confidence: float
    similarity: float
    distance: float
    threshold: float
    processing_time: float


class DetectionResponse(BaseModel):
    """Response model for detection endpoint."""
    face_count: int
    faces: list


class EmbeddingResponse(BaseModel):
    """Response model for embedding endpoint."""
    embedding: list
    dimension: int
    model: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str


class InfoResponse(BaseModel):
    """Response model for API info."""
    name: str
    version: str
    detector: str
    embedding_model: str
    threshold: float


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array."""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return np.array(image.convert('RGB'))


def load_upload_image(image_data: bytes) -> np.ndarray:
    """Load image from uploaded bytes."""
    image = Image.open(BytesIO(image_data))
    return np.array(image.convert('RGB'))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=__version__,
    )


@app.get("/info", response_model=InfoResponse)
async def api_info():
    """API information endpoint."""
    return InfoResponse(
        name="FaceVerify API",
        version=__version__,
        detector=config.detector_backend,
        embedding_model=config.embedding_model,
        threshold=config.threshold,
    )


@app.post("/verify", response_model=VerifyResponse)
async def verify_faces_upload(
    image1: UploadFile = File(..., description="First face image"),
    image2: UploadFile = File(..., description="Second face image"),
    threshold: Optional[float] = Form(None, description="Custom threshold (0-1)"),
):
    """
    Verify if two uploaded images contain the same person.
    
    Args:
        image1: First image file
        image2: Second image file
        threshold: Optional verification threshold (0-1)
        
    Returns:
        VerifyResponse with verification result
    """
    try:
        # Read and decode images
        img1_data = await image1.read()
        img2_data = await image2.read()
        
        img1 = load_upload_image(img1_data)
        img2 = load_upload_image(img2_data)
        
        # Update threshold if provided
        if threshold is not None:
            if not 0 <= threshold <= 1:
                raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")
            verifier.threshold = threshold
        
        # Perform verification
        result = verifier.verify(img1, img2)
        
        return VerifyResponse(
            verified=result.verified,
            confidence=result.confidence,
            similarity=result.similarity,
            distance=result.distance,
            threshold=result.threshold,
            processing_time=result.processing_time,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class Base64VerifyRequest(BaseModel):
    """Request model for base64 verification."""
    image1_base64: str
    image2_base64: str
    threshold: Optional[float] = None


@app.post("/verify/base64", response_model=VerifyResponse)
async def verify_faces_base64(request: Base64VerifyRequest):
    """
    Verify if two base64-encoded images contain the same person.
    
    Args:
        request: Request with base64 encoded images
        
    Returns:
        VerifyResponse with verification result
    """
    try:
        # Decode images
        img1 = decode_base64_image(request.image1_base64)
        img2 = decode_base64_image(request.image2_base64)
        
        # Update threshold if provided
        if request.threshold is not None:
            if not 0 <= request.threshold <= 1:
                raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")
            verifier.threshold = request.threshold
        
        # Perform verification
        result = verifier.verify(img1, img2)
        
        return VerifyResponse(
            verified=result.verified,
            confidence=result.confidence,
            similarity=result.similarity,
            distance=result.distance,
            threshold=result.threshold,
            processing_time=result.processing_time,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/detect", response_model=DetectionResponse)
async def detect_faces(
    image: UploadFile = File(..., description="Image to detect faces in"),
):
    """
    Detect faces in an uploaded image.
    
    Args:
        image: Image file
        
    Returns:
        DetectionResponse with detected faces
    """
    try:
        # Read and decode image
        img_data = await image.read()
        img = load_upload_image(img_data)
        
        # Detect faces (get all faces)
        detections = verifier.detect_faces(img, return_all=True)
        
        # Handle single detection or list
        if not isinstance(detections, list):
            detections = [detections] if detections else []
        
        faces = []
        for det in detections:
            faces.append({
                "bounding_box": det.bounding_box.to_tuple(),
                "confidence": det.confidence,
            })
        
        return DetectionResponse(
            face_count=len(faces),
            faces=faces,
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/embedding", response_model=EmbeddingResponse)
async def extract_embedding(
    image: UploadFile = File(..., description="Image containing a face"),
):
    """
    Extract face embedding from an uploaded image.
    
    Args:
        image: Image file containing a face
        
    Returns:
        EmbeddingResponse with the embedding vector
    """
    try:
        # Read and decode image
        img_data = await image.read()
        img = load_upload_image(img_data)
        
        # Extract embedding
        result = verifier.extract_embedding(img)
        
        return EmbeddingResponse(
            embedding=result.embedding.tolist(),
            dimension=result.dimension,
            model=result.model_name,
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("Starting FaceVerify API server...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
