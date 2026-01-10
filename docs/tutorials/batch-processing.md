# Batch Processing Tutorial

Process multiple image pairs efficiently.

---

## Prerequisites

- FaceVerify installed
- CSV file with image pairs OR list of image pairs

---

## Method 1: Python API

### Using verify_batch()

```python
from faceverify import FaceVerifier

verifier = FaceVerifier()

# Define pairs
pairs = [
    ("person1_a.jpg", "person1_b.jpg"),
    ("person2_a.jpg", "person2_b.jpg"),
    ("person1_a.jpg", "person2_a.jpg"),
]

# Process all pairs
results = verifier.verify_batch(pairs)

# Analyze results
for (img1, img2), result in zip(pairs, results):
    status = "MATCH" if result.verified else "NO MATCH"
    print(f"{img1} vs {img2}: {status} ({result.confidence:.2%})")
```

### With Progress Bar

```python
from tqdm import tqdm

pairs = [...]  # Your pairs

results = []
for pair in tqdm(pairs, desc="Verifying"):
    result = verifier.verify(pair[0], pair[1])
    results.append(result)
```

---

## Method 2: CLI

### CSV Format

Create `pairs.csv`:

```csv
image1,image2
images/person1_a.jpg,images/person1_b.jpg
images/person2_a.jpg,images/person2_b.jpg
images/person1_a.jpg,images/person2_a.jpg
```

### Run Batch Command

```bash
python -m faceverify batch pairs.csv -o results.json
```

### With Options

```bash
python -m faceverify batch pairs.csv \
    -o results.json \
    --threshold 0.70 \
    --parallel 4
```

---

## Method 3: Custom Script

### Full Example

```python
"""Batch face verification with detailed reporting."""

import csv
import json
from pathlib import Path
from faceverify import FaceVerifier
from faceverify.config import VerifierConfig


def load_pairs(csv_path):
    """Load image pairs from CSV file."""
    pairs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row['image1'], row['image2']))
    return pairs


def validate_pairs(pairs):
    """Check if all image files exist."""
    valid_pairs = []
    invalid_pairs = []
    
    for img1, img2 in pairs:
        if Path(img1).exists() and Path(img2).exists():
            valid_pairs.append((img1, img2))
        else:
            invalid_pairs.append((img1, img2))
    
    return valid_pairs, invalid_pairs


def process_batch(pairs, config=None):
    """Process all pairs and return results."""
    verifier = FaceVerifier(config)
    
    results = []
    for img1, img2 in pairs:
        try:
            result = verifier.verify(img1, img2)
            results.append({
                'image1': img1,
                'image2': img2,
                'verified': result.verified,
                'confidence': result.confidence,
                'similarity': result.similarity,
                'error': None,
            })
        except Exception as e:
            results.append({
                'image1': img1,
                'image2': img2,
                'verified': False,
                'confidence': 0.0,
                'similarity': 0.0,
                'error': str(e),
            })
    
    return results


def generate_report(results):
    """Generate summary report."""
    total = len(results)
    verified = sum(1 for r in results if r['verified'])
    errors = sum(1 for r in results if r['error'])
    not_verified = total - verified - errors
    
    avg_confidence = sum(r['confidence'] for r in results) / total if total else 0
    
    print("\n" + "=" * 50)
    print("  Batch Verification Report")
    print("=" * 50)
    print(f"  Total pairs:     {total}")
    print(f"  Verified:        {verified} ({verified/total*100:.1f}%)")
    print(f"  Not verified:    {not_verified} ({not_verified/total*100:.1f}%)")
    print(f"  Errors:          {errors} ({errors/total*100:.1f}%)")
    print(f"  Avg confidence:  {avg_confidence:.2%}")
    print("=" * 50 + "\n")


def main():
    # Load pairs
    pairs = load_pairs("pairs.csv")
    print(f"Loaded {len(pairs)} pairs")
    
    # Validate
    valid, invalid = validate_pairs(pairs)
    if invalid:
        print(f"Warning: {len(invalid)} pairs have missing files")
    
    # Configure
    config = VerifierConfig(
        detector_backend="opencv",
        threshold=0.65,
    )
    
    # Process
    print("Processing...")
    results = process_batch(valid, config)
    
    # Save results
    with open("results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Report
    generate_report(results)


if __name__ == "__main__":
    main()
```

---

## Performance Tips

> [!TIP]
> For large batches:
> - Use parallel processing: `--parallel 4`
> - Enable GPU: `enable_gpu=True`
> - Use faster detector: `detector_backend="opencv"`

---

## Next Steps

- [Face Identification](identification.md) - 1:N matching
- [Performance Guide](../guides/performance.md) - Optimization
