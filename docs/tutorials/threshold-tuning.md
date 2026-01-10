# Threshold Tuning Tutorial

Optimize verification threshold for your use case.

---

## Understanding Thresholds

The threshold determines when two faces are considered a match:

```
similarity >= threshold  -->  VERIFIED
similarity <  threshold  -->  NOT VERIFIED
```

---

## Trade-offs

| Threshold | False Accepts | False Rejects | Use Case |
|-----------|---------------|---------------|----------|
| High (0.75+) | Low | High | High security |
| Medium (0.60-0.70) | Medium | Medium | General use |
| Low (0.50-0.55) | High | Low | Loose matching |

> [!IMPORTANT]
> There is no perfect threshold. You must balance false accepts vs false rejects based on your requirements.

---

## Method 1: Manual Testing

### Test with Known Pairs

```python
from faceverify import FaceVerifier

verifier = FaceVerifier()

# Same person pairs (should verify)
same_person_pairs = [
    ("person1_a.jpg", "person1_b.jpg"),
    ("person2_a.jpg", "person2_b.jpg"),
]

# Different person pairs (should not verify)
different_person_pairs = [
    ("person1_a.jpg", "person2_a.jpg"),
    ("person1_a.jpg", "person3_a.jpg"),
]

# Collect similarities
same_similarities = []
for img1, img2 in same_person_pairs:
    result = verifier.verify(img1, img2)
    same_similarities.append(result.similarity)

diff_similarities = []
for img1, img2 in different_person_pairs:
    result = verifier.verify(img1, img2)
    diff_similarities.append(result.similarity)

print(f"Same person similarities: {same_similarities}")
print(f"  Min: {min(same_similarities):.4f}")
print(f"  Max: {max(same_similarities):.4f}")

print(f"\nDifferent person similarities: {diff_similarities}")
print(f"  Min: {min(diff_similarities):.4f}")
print(f"  Max: {max(diff_similarities):.4f}")
```

### Find Optimal Threshold

```python
# Threshold should be:
# - Above max different person similarity (avoid false accepts)
# - Below min same person similarity (avoid false rejects)

max_diff = max(diff_similarities)
min_same = min(same_similarities)

if max_diff < min_same:
    optimal = (max_diff + min_same) / 2
    print(f"\nOptimal threshold range: {max_diff:.4f} - {min_same:.4f}")
    print(f"Suggested threshold: {optimal:.4f}")
else:
    print("\nWarning: Overlapping distributions")
    print("Consider using better quality images or different model")
```

---

## Method 2: Grid Search

```python
"""Find optimal threshold using grid search."""

from faceverify import FaceVerifier
from faceverify.config import VerifierConfig


def evaluate_threshold(verifier, same_pairs, diff_pairs, threshold):
    """Evaluate a threshold value."""
    verifier.threshold = threshold
    
    # Test same person pairs
    true_positives = 0
    false_negatives = 0
    for img1, img2 in same_pairs:
        result = verifier.verify(img1, img2)
        if result.verified:
            true_positives += 1
        else:
            false_negatives += 1
    
    # Test different person pairs
    true_negatives = 0
    false_positives = 0
    for img1, img2 in diff_pairs:
        result = verifier.verify(img1, img2)
        if not result.verified:
            true_negatives += 1
        else:
            false_positives += 1
    
    total = len(same_pairs) + len(diff_pairs)
    accuracy = (true_positives + true_negatives) / total
    
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
    }


# Your test data
same_pairs = [...]  # Same person pairs
diff_pairs = [...]  # Different person pairs

verifier = FaceVerifier()

# Test range of thresholds
thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
results = []

for t in thresholds:
    result = evaluate_threshold(verifier, same_pairs, diff_pairs, t)
    results.append(result)
    print(f"Threshold {t:.2f}: Accuracy {result['accuracy']:.2%}")

# Find best
best = max(results, key=lambda x: x["accuracy"])
print(f"\nBest threshold: {best['threshold']:.2f} (Accuracy: {best['accuracy']:.2%})")
```

---

## Method 3: ROC Analysis

```python
"""ROC curve analysis for threshold tuning."""

import numpy as np
from faceverify import FaceVerifier


def collect_scores(verifier, same_pairs, diff_pairs):
    """Collect similarity scores."""
    same_scores = []
    for img1, img2 in same_pairs:
        result = verifier.verify(img1, img2)
        same_scores.append(result.similarity)
    
    diff_scores = []
    for img1, img2 in diff_pairs:
        result = verifier.verify(img1, img2)
        diff_scores.append(result.similarity)
    
    return same_scores, diff_scores


def calculate_rates(same_scores, diff_scores, threshold):
    """Calculate TPR and FPR at threshold."""
    tpr = sum(1 for s in same_scores if s >= threshold) / len(same_scores)
    fpr = sum(1 for s in diff_scores if s >= threshold) / len(diff_scores)
    return tpr, fpr


# Collect scores
verifier = FaceVerifier()
same_scores, diff_scores = collect_scores(verifier, same_pairs, diff_pairs)

# Calculate ROC points
thresholds = np.arange(0.3, 0.95, 0.05)
roc_points = []

for t in thresholds:
    tpr, fpr = calculate_rates(same_scores, diff_scores, t)
    roc_points.append((fpr, tpr, t))

# Find threshold closest to (0, 1)
best_threshold = min(
    roc_points,
    key=lambda p: np.sqrt(p[0]**2 + (1-p[1])**2)
)[2]

print(f"Optimal threshold (ROC): {best_threshold:.4f}")
```

---

## Recommendations by Use Case

### High Security (Banking, Access Control)

```python
config = VerifierConfig(
    threshold=0.75,
    detector_backend="retinaface",  # Most accurate
)
```

> [!CAUTION]
> High threshold may reject legitimate users. Consider multi-factor authentication.

### Standard Applications

```python
config = VerifierConfig(
    threshold=0.65,  # Default
    detector_backend="mtcnn",
)
```

### User Convenience (Photo Organization)

```python
config = VerifierConfig(
    threshold=0.55,
    detector_backend="opencv",  # Faster
)
```

> [!WARNING]
> Low threshold may group different people together.

---

## Next Steps

- [Configuration Guide](../guides/configuration.md) - All settings
- [Performance Guide](../guides/performance.md) - Speed optimization
