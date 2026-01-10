# Command Line Interface

FaceVerify CLI reference.

---

## Installation

The CLI is installed automatically with the package:

```bash
pip install faceverify-sdk
```

---

## Usage

```bash
python -m faceverify [command] [options]
```

> [!IMPORTANT]
> On Windows, always use `python -m faceverify` instead of `faceverify` directly due to PATH issues.

---

## Commands

### verify

Verify if two faces belong to the same person.

```bash
python -m faceverify verify <image1> <image2> [options]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `image1` | Path to first image |
| `image2` | Path to second image |

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--threshold` | `-t` | auto | Similarity threshold (0.0-1.0) |
| `--detector` | `-d` | `opencv` | Detection backend |
| `--embedding` | `-e` | `facenet` | Embedding model |
| `--metric` | `-m` | `cosine` | Similarity metric |
| `--json` | | | Output as JSON |

#### Detector Options

- `opencv` - Fast, built-in (default)
- `mtcnn` - Accurate, requires `pip install mtcnn`
- `retinaface` - Most accurate, requires `pip install retinaface`
- `mediapipe` - Fast, requires `pip install mediapipe`

#### Embedding Options

- `facenet` - Facenet512 via DeepFace (default)
- `arcface` - ArcFace model
- `vggface` - VGG-Face model

#### Metric Options

- `cosine` - Cosine similarity (default)
- `euclidean` - Euclidean distance
- `manhattan` - Manhattan distance

#### Examples

```bash
# Basic verification
python -m faceverify verify person1.jpg person2.jpg

# Custom threshold
python -m faceverify verify person1.jpg person2.jpg -t 0.70

# Use MTCNN detector
python -m faceverify verify person1.jpg person2.jpg -d mtcnn

# JSON output
python -m faceverify verify person1.jpg person2.jpg --json
```

#### Output

```
==================================================
  Face Verification Result
==================================================
  Status:      VERIFIED
  Confidence:  92.31%
  Similarity:  0.9231
  Threshold:   0.6500
==================================================
```

---

### detect

Detect faces in an image.

```bash
python -m faceverify detect <image> [options]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `image` | Path to input image |

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | | Output directory for extracted faces |
| `--detector` | `-d` | `opencv` | Detection backend |
| `--json` | | | Output as JSON |

#### Examples

```bash
# Detect faces
python -m faceverify detect group_photo.jpg

# Save detected faces to directory
python -m faceverify detect group_photo.jpg -o ./extracted_faces/

# JSON output
python -m faceverify detect group_photo.jpg --json
```

#### Output

```
Detected 3 face(s) in group_photo.jpg
  Face 1: confidence=98.50%
  Face 2: confidence=97.20%
  Face 3: confidence=95.80%
```

---

### batch

Process multiple image pairs from a CSV file.

```bash
python -m faceverify batch <input.csv> [options]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `input.csv` | CSV file with `image1,image2` columns |

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `results.json` | Output file path |
| `--threshold` | `-t` | auto | Similarity threshold |
| `--parallel` | | `1` | Number of parallel workers |

#### CSV Format

```csv
image1,image2
test_images/person1_a.jpg,test_images/person1_b.jpg
test_images/person1_a.jpg,test_images/person2.jpg
```

#### Examples

```bash
# Process pairs
python -m faceverify batch pairs.csv

# Custom output and parallel processing
python -m faceverify batch pairs.csv -o results.json --parallel 4
```

#### Output

```
Processing 10 image pairs...
100%|████████████████████████████████| 10/10 [00:15<00:00]

Results saved to: results.json
  Total pairs:    10
  Verified:       4
  Not verified:   5
  Errors:         1
```

---

### info

Display system and library information.

```bash
python -m faceverify info
```

#### Output

```
FaceVerify v1.0.0rc1
========================================
Python:        3.11.0
Platform:      Windows 10
Architecture:  AMD64

Available Backends:
  OpenCV:      4.8.0
  TensorFlow:  2.15.0
  GPU:         1 device(s) available
  ONNX:        1.15.0

Detection Backends:
  MTCNN:       Available
  MediaPipe:   Not installed (pip install mediapipe)
  OpenCV:      Built-in (always available)
```

---

## Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | `-v` | Show version and exit |
| `--verbose` | | Enable verbose output |
| `--config` | | Path to YAML config file |

#### Examples

```bash
# Show version
python -m faceverify --version

# Use config file
python -m faceverify --config myconfig.yaml verify img1.jpg img2.jpg
```

---

## Exit Codes

| Code | Description |
|------|-------------|
| `0` | Success (or verified for verify command) |
| `1` | Error or not verified |
| `130` | Interrupted by user (Ctrl+C) |

> [!TIP]
> Use exit codes in scripts to check verification results:
> ```bash
> if python -m faceverify verify a.jpg b.jpg; then
>     echo "Same person"
> else
>     echo "Different people"
> fi
>
