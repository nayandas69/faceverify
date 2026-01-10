# tests_real.py
from faceverify import FaceVerifier

verifier = FaceVerifier()

# Test same person
result = verifier.verify("test_images/person1_a.jpg", "test_images/person1_b.jpg")
print(f"Same person test:")
print(f"  Verified: {result.verified}")
print(f"  Confidence: {result.confidence:.2%}")
print(f"  Similarity: {result.similarity_score:.4f}")

# Test different people
result2 = verifier.verify("test_images/person1_a.jpg", "test_images/person2.jpg")
print(f"\nDifferent person test:")
print(f"  Verified: {result2.verified}")
print(f"  Confidence: {result2.confidence:.2%}")
