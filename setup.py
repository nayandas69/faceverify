#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FaceVerify Setup Script
=======================

This file provides backward compatibility for installations using:
    python setup.py install

For modern installations, use:
    pip install -e .

Note: pyproject.toml is the primary configuration file.
"""

import os
import sys
from pathlib import Path

from setuptools import find_packages, setup

# ==============================================================================
# Package Metadata
# ==============================================================================

NAME = "faceverify-sdk"
VERSION = "1.0.0rc1"
DESCRIPTION = "A modular, open-source face verification SDK for Python"
AUTHOR = "nayandas69"
AUTHOR_EMAIL = "nayanchandradas@hotmail.com"
URL = "https://github.com/nayandas69/faceverify"
LICENSE = "MIT"
PYTHON_REQUIRES = ">=3.8"

# ==============================================================================
# Long Description
# ==============================================================================

HERE = Path(__file__).parent.resolve()

try:
    LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION

# ==============================================================================
# Dependencies
# ==============================================================================

INSTALL_REQUIRES = [
    "numpy>=1.21.0",
    "opencv-python>=4.5.0",
    "Pillow>=8.0.0",
    "scikit-learn>=1.0.0",
    "deepface>=0.0.79",
    "tf-keras>=2.15.0",
    "pyyaml>=6.0",
    "pydantic>=2.0.0",
    "tqdm>=4.64.0",
    "structlog>=23.0.0",
]

EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
    ],
    "docs": [
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=1.0.0",
    ],
    "gpu": [
        "tensorflow>=2.10.0",
        "onnxruntime-gpu>=1.12.0",
    ],
    "api": [
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "python-multipart>=0.0.6",
    ],
    "detection": [
        "mtcnn>=0.1.1",
        "mediapipe>=0.10.0",
        "retinaface>=0.0.13",
    ],
}

EXTRAS_REQUIRE["all"] = list(
    set(
        sum(
            [
                EXTRAS_REQUIRE["dev"],
                EXTRAS_REQUIRE["docs"],
                EXTRAS_REQUIRE["api"],
                EXTRAS_REQUIRE["detection"],
            ],
            [],
        )
    )
)

# ==============================================================================
# Classifiers
# ==============================================================================

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Recognition",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

# ==============================================================================
# Keywords
# ==============================================================================

KEYWORDS = [
    "face-recognition",
    "face-verification",
    "face-detection",
    "deep-learning",
    "computer-vision",
    "biometrics",
    "identity-verification",
    "facenet",
    "deepface",
    "opencv",
]

# ==============================================================================
# Entry Points
# ==============================================================================

ENTRY_POINTS = {
    "console_scripts": [
        "faceverify=faceverify.cli:main",
    ],
}

# ==============================================================================
# Setup
# ==============================================================================


def main():
    """Run setup."""
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        license=LICENSE,
        python_requires=PYTHON_REQUIRES,
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
        entry_points=ENTRY_POINTS,
        zip_safe=False,
        project_urls={
            "Documentation": "https://github.com/nayandas69/faceverify#readme",
            "Source": "https://github.com/nayandas69/faceverify",
            "Changelog": "https://github.com/nayandas69/faceverify/blob/main/CHANGELOG.md",
            "Bug Tracker": "https://github.com/nayandas69/faceverify/issues",
        },
    )


if __name__ == "__main__":
    main()
