"""
FaceVerify CLI - Command Line Interface for Face Verification.

This module provides a comprehensive CLI for the FaceVerify SDK,
supporting face verification, batch processing, and configuration management.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from faceverify import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="faceverify",
        description="FaceVerify - Production-ready Face Verification SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  faceverify verify image1.jpg image2.jpg
  faceverify verify image1.jpg image2.jpg --threshold 0.7
  faceverify detect image.jpg --output faces/
  faceverify batch pairs.csv --output results.json

For more information, visit: https://github.com/nayandas69/faceverify
        """,
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"FaceVerify v{__version__}",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (YAML)",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
    )

    # Verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify if two faces belong to the same person",
    )
    verify_parser.add_argument(
        "image1",
        type=str,
        help="Path to the first face image",
    )
    verify_parser.add_argument(
        "image2",
        type=str,
        help="Path to the second face image",
    )
    verify_parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold for verification (0.0-1.0)",
    )
    verify_parser.add_argument(
        "-d",
        "--detector",
        type=str,
        choices=["mtcnn", "retinaface", "mediapipe", "opencv"],
        default="opencv",  # Default to opencv (no extra install needed)
        help="Face detection backend",
    )
    verify_parser.add_argument(
        "-e",
        "--embedding",
        type=str,
        choices=["facenet", "arcface", "vggface"],
        default="facenet",
        help="Face embedding model",
    )
    verify_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["cosine", "euclidean", "manhattan"],
        default="cosine",
        help="Similarity metric",
    )
    verify_parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )

    # Detect command
    detect_parser = subparsers.add_parser(
        "detect",
        help="Detect faces in an image",
    )
    detect_parser.add_argument(
        "image",
        type=str,
        help="Path to the input image",
    )
    detect_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory for extracted faces",
    )
    detect_parser.add_argument(
        "-d",
        "--detector",
        type=str,
        choices=["mtcnn", "retinaface", "mediapipe", "opencv"],
        default="opencv",  # Default to opencv
        help="Face detection backend",
    )
    detect_parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Process multiple image pairs from a CSV file",
    )
    batch_parser.add_argument(
        "input_file",
        type=str,
        help="Path to CSV file with image pairs (columns: image1, image2)",
    )
    batch_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results.json",
        help="Output file path for results",
    )
    batch_parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold for verification",
    )
    batch_parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers",
    )

    # Info command
    subparsers.add_parser(
        "info",
        help="Display system and library information",
    )

    return parser


def cmd_verify(args: argparse.Namespace) -> int:
    """Execute the verify command."""
    import json

    from faceverify import FaceVerifier
    from faceverify.config import VerifierConfig  # Use correct class name

    # Validate image paths
    image1_path = Path(args.image1)
    image2_path = Path(args.image2)

    if not image1_path.exists():
        print(f"Error: Image not found: {args.image1}", file=sys.stderr)
        return 1

    if not image2_path.exists():
        print(f"Error: Image not found: {args.image2}", file=sys.stderr)
        return 1

    config_kwargs = {
        "detector_backend": args.detector,
        "embedding_model": args.embedding,
        "similarity_metric": args.metric,
    }
    if args.threshold is not None:
        config_kwargs["threshold"] = args.threshold

    config = VerifierConfig(**config_kwargs)

    # Initialize verifier and run verification
    if args.verbose:
        print(f"Initializing FaceVerifier with config: {config}")

    verifier = FaceVerifier(config=config)
    result = verifier.verify(str(image1_path), str(image2_path))

    # Output result
    if args.json:
        output = {
            "verified": result.verified,
            "confidence": result.confidence,
            "similarity": result.similarity,
            "threshold": result.threshold,
            "image1": str(image1_path),
            "image2": str(image2_path),
        }
        print(json.dumps(output, indent=2))
    else:
        status = "VERIFIED" if result.verified else "NOT VERIFIED"
        print(f"\n{'='*50}")
        print(f"  Face Verification Result")
        print(f"{'='*50}")
        print(f"  Status:      {status}")
        print(f"  Confidence:  {result.confidence:.2%}")
        print(f"  Similarity:  {result.similarity:.4f}")
        print(f"  Threshold:   {result.threshold:.4f}")
        print(f"{'='*50}\n")

    return 0 if result.verified else 1


def cmd_detect(args: argparse.Namespace) -> int:
    """Execute the detect command."""
    import json

    import cv2
    import numpy as np

    from faceverify.detection import create_detector

    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        return 1

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image: {args.image}", file=sys.stderr)
        return 1

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    detector = create_detector(args.detector)
    faces = detector.detect(image_rgb)

    if args.json:
        output = {
            "image": str(image_path),
            "faces_detected": len(faces),
            "faces": [
                {
                    "bbox": (
                        face.bbox.tolist()
                        if isinstance(face.bbox, np.ndarray)
                        else face.bbox
                    ),
                    "confidence": float(face.confidence),
                }
                for face in faces
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nDetected {len(faces)} face(s) in {image_path.name}")
        for i, face in enumerate(faces):
            print(f"  Face {i+1}: confidence={face.confidence:.2%}")

    # Save extracted faces if output directory specified
    if args.output and faces:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = map(int, face.bbox[:4])
            face_crop = image[y1:y2, x1:x2]
            output_path = output_dir / f"{image_path.stem}_face_{i+1}.jpg"
            cv2.imwrite(str(output_path), face_crop)
            if not args.json:
                print(f"  Saved: {output_path}")

    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Execute the batch command."""
    import csv
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tqdm import tqdm

    from faceverify import FaceVerifier
    from faceverify.config import VerifierConfig

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        return 1

    # Read image pairs from CSV
    pairs = []
    with open(input_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "image1" in row and "image2" in row:
                pairs.append((row["image1"], row["image2"]))

    if not pairs:
        print("Error: No valid image pairs found in CSV", file=sys.stderr)
        return 1

    print(f"Processing {len(pairs)} image pairs...")

    # Create configuration
    config_kwargs = {}
    if args.threshold is not None:
        config_kwargs["threshold"] = args.threshold

    config = VerifierConfig(**config_kwargs)
    verifier = FaceVerifier(config=config)

    # Process pairs
    results = []

    def process_pair(pair):
        image1, image2 = pair
        try:
            result = verifier.verify(image1, image2)
            return {
                "image1": image1,
                "image2": image2,
                "verified": result.verified,
                "confidence": result.confidence,
                "similarity": result.similarity,
                "error": None,
            }
        except Exception as e:
            return {
                "image1": image1,
                "image2": image2,
                "verified": False,
                "confidence": 0.0,
                "similarity": 0.0,
                "error": str(e),
            }

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(process_pair, pair): pair for pair in pairs}
        for future in tqdm(as_completed(futures), total=len(pairs)):
            results.append(future.result())

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    verified_count = sum(1 for r in results if r["verified"])
    error_count = sum(1 for r in results if r["error"])

    print(f"\nResults saved to: {output_path}")
    print(f"  Total pairs:    {len(results)}")
    print(f"  Verified:       {verified_count}")
    print(f"  Not verified:   {len(results) - verified_count - error_count}")
    print(f"  Errors:         {error_count}")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command."""
    import platform

    from faceverify import __version__

    print(f"\nFaceVerify v{__version__}")
    print("=" * 40)
    print(f"Python:        {platform.python_version()}")
    print(f"Platform:      {platform.system()} {platform.release()}")
    print(f"Architecture:  {platform.machine()}")

    # Check available backends
    print("\nAvailable Backends:")

    try:
        import cv2

        print(f"  OpenCV:      {cv2.__version__}")
    except ImportError:
        print("  OpenCV:      Not installed")

    try:
        import tensorflow as tf

        print(f"  TensorFlow:  {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        print(f"  GPU:         {len(gpus)} device(s) available")
    except ImportError:
        print("  TensorFlow:  Not installed")

    try:
        import onnxruntime as ort

        print(f"  ONNX:        {ort.__version__}")
    except ImportError:
        print("  ONNX:        Not installed")

    print("\nDetection Backends:")

    try:
        import mtcnn

        print(f"  MTCNN:       Available")
    except ImportError:
        print(f"  MTCNN:       Not installed (pip install mtcnn)")

    try:
        import mediapipe

        print(f"  MediaPipe:   Available")
    except ImportError:
        print(f"  MediaPipe:   Not installed (pip install mediapipe)")

    print(f"  OpenCV:      Built-in (always available)")

    print()
    return 0


def main(argv: Optional[list] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle no command case
    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to command handlers
    commands = {
        "verify": cmd_verify,
        "detect": cmd_detect,
        "batch": cmd_batch,
        "info": cmd_info,
    }

    handler = commands.get(args.command)
    if handler:
        try:
            return handler(args)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 130
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if hasattr(args, "verbose") and args.verbose:
                import traceback

                traceback.print_exc()
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
