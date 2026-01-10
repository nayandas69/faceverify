"""Command-line interface for FaceVerify."""

import argparse
import sys
from pathlib import Path


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="FaceVerify - Face Verification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  faceverify verify image1.jpg image2.jpg
  faceverify verify image1.jpg image2.jpg --threshold 0.7
  faceverify detect image.jpg --output faces/
  faceverify identify query.jpg --database ./known_faces/
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify two faces")
    verify_parser.add_argument("image1", help="First image path")
    verify_parser.add_argument("image2", help="Second image path")
    verify_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.65,
        help="Verification threshold (default: 0.65)",
    )
    verify_parser.add_argument(
        "--detector",
        "-d",
        default="mtcnn",
        choices=["mtcnn", "retinaface", "mediapipe", "opencv"],
        help="Face detector backend",
    )
    verify_parser.add_argument(
        "--model",
        "-m",
        default="facenet",
        choices=["facenet", "arcface", "vggface"],
        help="Embedding model",
    )

    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect faces in image")
    detect_parser.add_argument("image", help="Image path")
    detect_parser.add_argument(
        "--output",
        "-o",
        help="Output directory for detected faces",
    )
    detect_parser.add_argument(
        "--detector",
        "-d",
        default="mtcnn",
        help="Face detector backend",
    )

    # Identify command
    identify_parser = subparsers.add_parser(
        "identify", help="Identify face against database"
    )
    identify_parser.add_argument("query", help="Query image path")
    identify_parser.add_argument(
        "--database",
        "-db",
        required=True,
        help="Database directory",
    )
    identify_parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of top matches",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "verify":
        _run_verify(args)
    elif args.command == "detect":
        _run_detect(args)
    elif args.command == "identify":
        _run_identify(args)


def _run_verify(args):
    """Run verification command."""
    from faceverify import FaceVerifier
    from faceverify.config import VerifierConfig

    config = VerifierConfig(
        detector_backend=args.detector,
        embedding_model=args.model,
        threshold=args.threshold,
    )

    verifier = FaceVerifier(config)

    try:
        result = verifier.verify(args.image1, args.image2)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def _run_detect(args):
    """Run detection command."""
    from faceverify import FaceVerifier
    from faceverify.config import VerifierConfig
    from faceverify.utils.image import save_image

    config = VerifierConfig(detector_backend=args.detector)
    verifier = FaceVerifier(config)

    try:
        detections = verifier.detect_faces(args.image, return_all=True)

        print(f"Detected {len(detections)} face(s)")

        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            for i, det in enumerate(detections):
                if det.face_image is not None:
                    face_path = output_dir / f"face_{i}.jpg"
                    save_image(det.face_image, str(face_path))
                    print(f"  Saved: {face_path}")

        for i, det in enumerate(detections):
            print(f"\nFace {i + 1}:")
            print(f"  Confidence: {det.confidence:.2%}")
            print(f"  Bounding box: {det.bounding_box.to_tuple()}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def _run_identify(args):
    """Run identification command."""
    from faceverify import FaceVerifier

    verifier = FaceVerifier()

    try:
        result = verifier.identify(
            args.query,
            args.database,
            top_k=args.top_k,
        )

        print(f"Query: {result.query_image}")
        print(f"Searched {result.total_candidates} candidates")
        print(f"Found {len(result.matches)} match(es)")
        print(f"Time: {result.processing_time:.2f}s\n")

        for i, match in enumerate(result.matches):
            print(f"Match {i + 1}:")
            print(f"  Identity: {match['identity']}")
            print(f"  Similarity: {match['similarity']:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
