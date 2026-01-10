#!/usr/bin/env python3
"""
Webcam Face Verification Example
=================================

This example demonstrates real-time face verification
using a webcam feed.

Usage:
    python webcam_verification.py --reference reference.jpg

Requirements:
    - OpenCV with video capture support (pip install opencv-python)
    - Webcam connected to the computer
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import cv2
except ImportError:
    print("Error: OpenCV not installed.")
    print("Install with: pip install opencv-python")
    sys.exit(1)

import numpy as np

from faceverify import FaceVerifier
from faceverify.config import VerifierConfig


def main():
    """Main function for webcam verification."""
    
    parser = argparse.ArgumentParser(
        description="Real-time face verification using webcam"
    )
    parser.add_argument(
        "--reference", "-r",
        required=True,
        help="Path to reference image for verification",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.65,
        help="Verification threshold (default: 0.65)",
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=0.5,
        help="Verification interval in seconds (default: 0.5)",
    )
    
    args = parser.parse_args()
    
    # Validate reference image
    if not Path(args.reference).exists():
        print(f"Error: Reference image not found: {args.reference}")
        sys.exit(1)
    
    print("=" * 50)
    print("FaceVerify - Webcam Verification")
    print("=" * 50)
    
    # Initialize verifier
    config = VerifierConfig(
        detector_backend="opencv",  # Fast for real-time
        embedding_model="facenet",
        threshold=args.threshold,
    )
    verifier = FaceVerifier(config)
    
    # Load reference image and extract embedding
    print(f"\nLoading reference image: {args.reference}")
    
    try:
        reference_embedding = verifier.extract_embedding(args.reference)
        print(f"Reference embedding extracted (dim: {reference_embedding.dimension})")
    except Exception as e:
        print(f"Error loading reference image: {e}")
        sys.exit(1)
    
    # Open webcam
    print(f"\nOpening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        print("Make sure a webcam is connected and not in use by another application.")
        sys.exit(1)
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Camera opened successfully")
    print("\nControls:")
    print("  Press 'q' to quit")
    print("  Press 's' to save current frame")
    print("  Press 'r' to reload reference image")
    print("-" * 50)
    
    frame_count = 0
    last_verification_time = 0.0
    last_result: Optional[Dict[str, Any]] = None
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform verification periodically
            if current_time - last_verification_time >= args.interval:
                try:
                    # Extract embedding from current frame
                    frame_embedding = verifier.extract_embedding(frame_rgb)
                    
                    # Compare with reference
                    similarity, distance = verifier.compare_embeddings(
                        reference_embedding.embedding,
                        frame_embedding.embedding,
                    )
                    
                    verified = similarity >= args.threshold
                    
                    # Try to get face bounding box for display
                    try:
                        detection = verifier.detect_faces(frame_rgb, return_all=False)
                        bbox = detection.bounding_box if detection else None
                    except Exception:
                        bbox = None
                    
                    last_result = {
                        "verified": verified,
                        "similarity": similarity,
                        "bbox": bbox,
                    }
                    
                except Exception as e:
                    last_result = {"error": str(e)[:50]}
                
                last_verification_time = current_time
            
            # Draw results on frame
            if last_result:
                if "error" in last_result:
                    # Draw error message
                    cv2.putText(
                        frame, f"No face detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2,
                    )
                else:
                    # Draw bounding box if available
                    bbox = last_result.get("bbox")
                    color = (0, 255, 0) if last_result["verified"] else (0, 0, 255)
                    
                    if bbox:
                        cv2.rectangle(
                            frame,
                            (bbox.x1, bbox.y1),
                            (bbox.x2, bbox.y2),
                            color, 2,
                        )
                        
                        # Draw status above bounding box
                        status = "VERIFIED" if last_result["verified"] else "NOT VERIFIED"
                        cv2.putText(
                            frame, status,
                            (bbox.x1, bbox.y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2,
                        )
                    
                    # Draw similarity score
                    similarity_text = f"Similarity: {last_result['similarity']:.2%}"
                    cv2.putText(
                        frame, similarity_text,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2,
                    )
                    
                    # Draw threshold
                    threshold_text = f"Threshold: {args.threshold:.2%}"
                    cv2.putText(
                        frame, threshold_text,
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1,
                    )
            
            # Draw frame counter
            cv2.putText(
                frame, f"Frame: {frame_count}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (128, 128, 128), 1,
            )
            
            # Display frame
            cv2.imshow("FaceVerify - Webcam (press 'q' to quit)", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"capture_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
            elif key == ord('r'):
                # Reload reference image
                try:
                    reference_embedding = verifier.extract_embedding(args.reference)
                    print(f"Reference image reloaded: {args.reference}")
                except Exception as e:
                    print(f"Error reloading reference: {e}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
