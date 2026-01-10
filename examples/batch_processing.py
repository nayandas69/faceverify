#!/usr/bin/env python3
"""
Batch Processing Example
========================

This example demonstrates how to process multiple face
verification pairs efficiently.

Usage:
    python batch_processing.py

Note:
    Update the image paths in generate_verification_pairs()
    to match your actual test images.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple

from faceverify import FaceVerifier
from faceverify.config import VerifierConfig


def generate_verification_pairs() -> List[Tuple[str, str]]:
    """
    Generate sample verification pairs.
    
    Update these paths to match your actual image files.
    """
    # Example pairs - update these to your actual image paths
    return [
        ("test_images/person1_a.jpg", "test_images/person1_b.jpg"),  # Same person
        ("test_images/person1_a.jpg", "test_images/person2.jpg"),    # Different people
    ]


def validate_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Validate that all image files exist."""
    valid_pairs = []
    
    for img1, img2 in pairs:
        if not Path(img1).exists():
            print(f"Warning: File not found: {img1}")
            continue
        if not Path(img2).exists():
            print(f"Warning: File not found: {img2}")
            continue
        valid_pairs.append((img1, img2))
    
    return valid_pairs


def main():
    """Main function demonstrating batch processing."""
    
    print("=" * 50)
    print("FaceVerify - Batch Processing Example")
    print("=" * 50)
    
    # Initialize verifier
    config = VerifierConfig(
        detector_backend="opencv",
        embedding_model="facenet",
    )
    verifier = FaceVerifier(config)
    
    # Get verification pairs
    pairs = generate_verification_pairs()
    
    # Validate pairs
    valid_pairs = validate_pairs(pairs)
    
    if not valid_pairs:
        print("\nNo valid image pairs found.")
        print("Please update the generate_verification_pairs() function")
        print("with paths to your actual test images.")
        sys.exit(1)
    
    print(f"\nProcessing {len(valid_pairs)} verification pairs...")
    print("-" * 50)
    
    # Process batch
    start_time = time.time()
    results = verifier.verify_batch(valid_pairs)
    total_time = time.time() - start_time
    
    # Display results
    for i, (pair, result) in enumerate(zip(valid_pairs, results)):
        print(f"\nPair {i + 1}:")
        print(f"  Image 1: {pair[0]}")
        print(f"  Image 2: {pair[1]}")
        
        if result.metadata.get("error"):
            print(f"  Error: {result.metadata['error']}")
        else:
            status = "MATCH" if result.verified else "NO MATCH"
            print(f"  Result: {status}")
            print(f"  Similarity: {result.similarity:.4f}")
            print(f"  Time: {result.processing_time:.3f}s")
    
    # Summary statistics
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    successful = [r for r in results if not r.metadata.get("error")]
    matches = sum(1 for r in successful if r.verified)
    
    print(f"Total pairs: {len(valid_pairs)}")
    print(f"Successful: {len(successful)}")
    print(f"Matches: {matches}")
    print(f"Non-matches: {len(successful) - matches}")
    print(f"Total time: {total_time:.2f}s")
    if valid_pairs:
        print(f"Average time per pair: {total_time / len(valid_pairs):.3f}s")


if __name__ == "__main__":
    main()
