# Contributing to FaceVerify

Thank you for your interest in contributing to FaceVerify! This document provides guidelines and instructions for contributing.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Before You Push](#before-you-push)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)

---

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

---

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/faceverify.git
cd faceverify
```

3. Add upstream remote:

```bash
git remote add upstream https://github.com/nayandas69/faceverify.git
```

---

## Development Setup

### Step 1: Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

- Linux/macOS: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

### Step 2: Install Development Dependencies

```bash
pip install -e ".[dev]"
```

Or using requirements:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### Step 3: Verify Installation

```bash
python -m faceverify --version
pytest tests/ -v
```

---

## Before You Push

> [!CAUTION]
> Always complete these steps before pushing any changes!

### 1. Format Your Code

```bash
black src/ tests/
```

### 2. Run Linting (Optional but Recommended)

```bash
flake8 src/ tests/
mypy src/
```

### 3. Run All Tests

```bash
pytest tests/ -v
```

### 4. Check Test Coverage

```bash
pytest tests/ --cov=src/faceverify --cov-report=term-missing
```

> [!WARNING]
> Pull requests with failing tests will not be merged. Always run tests locally first.

---

## Making Changes

### Step 1: Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:

- `feature/add-new-detector`
- `fix/similarity-calculation`
- `docs/update-readme`

### Step 2: Make Your Changes

- Follow the [Style Guide](#style-guide)
- Write tests for new functionality
- Update documentation if needed

### Step 3: Commit Your Changes

```bash
git add .
git commit -m "feat: add your feature description"
```

Use [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Description |
|--------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation changes |
| `test:` | Adding or updating tests |
| `refactor:` | Code refactoring |
| `chore:` | Maintenance tasks |
| `style:` | Formatting changes |
| `perf:` | Performance improvements |

### Step 4: Push to Your Fork

```bash
git push origin feature/your-feature-name
```

---

## Pull Request Process

1. Ensure all tests pass locally
2. Update CHANGELOG.md with your changes
3. Push to your fork
4. Create a Pull Request on GitHub
5. Fill out the PR template completely
6. Wait for review and address feedback

> [!NOTE]
> PRs are typically reviewed within 2-3 days. Please be patient!

---

## Style Guide

### Python Code

- Follow PEP 8 style guide
- Use type hints for function arguments and return values
- Maximum line length: 88 characters (Black default)
- Use descriptive variable and function names

### Formatting

All code must be formatted with Black:

```bash
black src/ tests/
```

### Docstrings

Use Google-style docstrings:

```python
def verify_faces(
    image1: np.ndarray,
    image2: np.ndarray,
    threshold: float = 0.65,
) -> VerificationResult:
    """
    Verify if two images contain the same person.
    
    Args:
        image1: First face image as RGB numpy array.
        image2: Second face image as RGB numpy array.
        threshold: Similarity threshold for positive match.
        
    Returns:
        VerificationResult containing the verification outcome.
        
    Raises:
        NoFaceDetectedError: If no face is found in either image.
        
    Example:
        >>> result = verify_faces(img1, img2)
        >>> print(f"Match: {result.verified}")
    """
    ...
```

### Imports

Order imports as follows:

1. Standard library imports
2. Third-party imports
3. Local imports

```python
# Standard library
import os
from typing import Optional, List

# Third-party
import numpy as np
import cv2

# Local
from faceverify.core import VerificationResult
from faceverify.utils import load_image
```

---

## Testing Guidelines

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names
- Test both success and failure cases

### Test File Naming

```
tests/
├── unit/
│   ├── test_similarity.py
│   ├── test_detection.py
│   └── test_decision.py
└── integration/
    └── test_full_pipeline.py
```

### Test Function Naming

```python
def test_cosine_similarity_identical_vectors():
    """Test that identical vectors have similarity of 1.0"""
    ...

def test_verify_raises_on_no_face():
    """Test that NoFaceDetectedError is raised when no face found"""
    ...
```

---

## Questions?

Feel free to open an issue for any questions or concerns!

- Issues: https://github.com/nayandas69/faceverify/issues
- Discussions: https://github.com/nayandas69/faceverify/discussions

---

Thank you for contributing to FaceVerify!
