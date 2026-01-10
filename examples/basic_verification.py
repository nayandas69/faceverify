#!/usr/bin/env python3
"""
Basic Face Verification Example
================================

This example demonstrates how to use FaceVerify for basic
face verification between two images.

Usage:
    python basic_verification.py image1.jpg image2.jpg
"""

import sys
from pathlib import Path

from faceverify import FaceVerifier
from faceverify.config import VerifierConfig


def main():
    """Main function demonstrating basic verification."""
    
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python basic_verification.py image1.jpg image2.jpg")
        print("\nPlease provide two image paths as arguments.")
        sys.exit(1)
    
    image1 = sys.argv[1]
    image2 = sys.argv[2]
    
    # Check if files exist
    if not Path(image1).exists():
        print(f"Error: File not found: {image1}")
        sys.exit(1)
    if not Path(image2).exists():
        print(f"Error: File not found: {image2}")
        sys.exit(1)
    
    print("=" * 50)
    print("FaceVerify - Basic Verification Example")
    print("=" * 50)
    
    # Method 1: Default configuration
    print("\n[Method 1] Using default configuration:")
    print("-" * 40)
    
    verifier = FaceVerifier()
    
    try:
        result = verifier.verify(image1, image2)
        
        print(f"Image 1: {image1}")
        print(f"Image 2: {image2}")
        print(f"\nResult: {'MATCH' if result.verified else 'NO MATCH'}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Similarity: {result.similarity:.4f}")
        print(f"Threshold: {result.threshold:.4f}")
        print(f"Processing time: {result.processing_time:.3f}s")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Method 2: Custom configuration with stricter threshold
    print("\n[Method 2] Using custom configuration:")
    print("-" * 40)
    
    config = VerifierConfig(
        detector_backend="opencv",
        embedding_model="facenet",
        similarity_metric="cosine",
        threshold=0.70,  # Stricter threshold
    )
    
    verifier_custom = FaceVerifier(config)
    
    print(f"Detector: {config.detector_backend}")
    print(f"Model: {config.embedding_model}")
    print(f"Metric: {config.similarity_metric}")
    print(f"Threshold: {config.threshold}")
    
    try:
        result_custom = verifier_custom.verify(image1, image2)
        print(f"\nResult: {'MATCH' if result_custom.verified else 'NO MATCH'}")
        print(f"Similarity: {result_custom.similarity:.4f}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Method 3: Extract embeddings separately
    print("\n[Method 3] Extract embeddings separately:")
    print("-" * 40)
    
    try:
        # This is useful when you want to store embeddings in a database
        embedding1 = verifier.extract_embedding(image1)
        embedding2 = verifier.extract_embedding(image2)
        
        print(f"Embedding 1 dimension: {embedding1.dimension}")
        print(f"Embedding 2 dimension: {embedding2.dimension}")
        print(f"Model used: {embedding1.model_name}")
        
        # Compare embeddings directly
        similarity, distance = verifier.compare_embeddings(
            embedding1.embedding,
            embedding2.embedding
        )
        
        print(f"Similarity: {similarity:.4f}")
        print(f"Distance: {distance:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
