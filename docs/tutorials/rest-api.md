# REST API Tutorial

Build a face verification REST API.

---

## Prerequisites

- FaceVerify installed
- Flask: `pip install flask`

---

## Basic API

### server.py

```python
"""Simple face verification REST API."""

from flask import Flask, request, jsonify
from faceverify import FaceVerifier
from faceverify.config import VerifierConfig
import tempfile
import os

app = Flask(__name__)

# Initialize verifier
config = VerifierConfig(
    detector_backend="opencv",
    threshold=0.65,
)
verifier = FaceVerifier(config)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route("/verify", methods=["POST"])
def verify():
    """
    Verify two face images.
    
    Expects multipart form with 'image1' and 'image2' files.
    """
    # Check files
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"error": "Missing image1 or image2"}), 400
    
    file1 = request.files["image1"]
    file2 = request.files["image2"]
    
    # Save to temp files
    temp_dir = tempfile.mkdtemp()
    try:
        path1 = os.path.join(temp_dir, "image1.jpg")
        path2 = os.path.join(temp_dir, "image2.jpg")
        
        file1.save(path1)
        file2.save(path2)
        
        # Verify
        result = verifier.verify(path1, path2)
        
        return jsonify({
            "verified": result.verified,
            "confidence": result.confidence,
            "similarity": result.similarity,
            "threshold": result.threshold,
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Cleanup
        if os.path.exists(path1):
            os.remove(path1)
        if os.path.exists(path2):
            os.remove(path2)
        os.rmdir(temp_dir)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
```

### Run Server

```bash
python server.py
```

### Test API

```bash
curl -X POST http://localhost:8000/verify \
  -F "image1=@person1.jpg" \
  -F "image2=@person2.jpg"
```

---

## Production API

### With Validation and Error Handling

```python
"""Production-ready face verification API."""

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from faceverify import FaceVerifier
from faceverify.config import VerifierConfig
import tempfile
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

# Allowed extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Initialize verifier
verifier = FaceVerifier(VerifierConfig(
    detector_backend="opencv",
    threshold=0.65,
))


def allowed_file(filename):
    """Check if file extension is allowed."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def validate_image(file):
    """Validate uploaded image."""
    if not file:
        raise ValueError("No file provided")
    
    if file.filename == "":
        raise ValueError("Empty filename")
    
    if not allowed_file(file.filename):
        raise ValueError(f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}")
    
    return True


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "faceverify-api",
    })


@app.route("/verify", methods=["POST"])
def verify():
    """Verify two face images."""
    temp_files = []
    
    try:
        # Validate inputs
        if "image1" not in request.files:
            return jsonify({"error": "Missing image1"}), 400
        if "image2" not in request.files:
            return jsonify({"error": "Missing image2"}), 400
        
        file1 = request.files["image1"]
        file2 = request.files["image2"]
        
        validate_image(file1)
        validate_image(file2)
        
        # Save to temp files
        temp_dir = tempfile.mkdtemp()
        
        path1 = os.path.join(temp_dir, secure_filename(file1.filename))
        path2 = os.path.join(temp_dir, secure_filename(file2.filename))
        
        file1.save(path1)
        file2.save(path2)
        
        temp_files = [path1, path2, temp_dir]
        
        # Get optional threshold
        threshold = request.form.get("threshold", type=float)
        if threshold:
            verifier.threshold = threshold
        
        # Verify
        result = verifier.verify(path1, path2)
        
        logger.info(f"Verification: verified={result.verified}, confidence={result.confidence:.2%}")
        
        return jsonify({
            "verified": result.verified,
            "confidence": round(result.confidence, 4),
            "similarity": round(result.similarity, 4),
            "threshold": result.threshold,
            "processing_time": round(result.processing_time, 3),
        })
    
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({"error": str(e)}), 400
    
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return jsonify({"error": "Verification failed"}), 500
    
    finally:
        # Cleanup temp files
        for path in temp_files[:-1]:  # Files
            if os.path.exists(path):
                os.remove(path)
        if temp_files and os.path.exists(temp_files[-1]):  # Directory
            os.rmdir(temp_files[-1])


@app.route("/detect", methods=["POST"])
def detect():
    """Detect faces in an image."""
    temp_path = None
    
    try:
        if "image" not in request.files:
            return jsonify({"error": "Missing image"}), 400
        
        file = request.files["image"]
        validate_image(file)
        
        # Save temp file
        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        file.save(temp_path)
        
        # Detect
        faces = verifier.detect_faces(temp_path, return_all=True)
        
        if not isinstance(faces, list):
            faces = [faces] if faces else []
        
        return jsonify({
            "faces_detected": len(faces),
            "faces": [
                {
                    "confidence": round(f.confidence, 4),
                    "bounding_box": {
                        "x": f.bounding_box.x,
                        "y": f.bounding_box.y,
                        "width": f.bounding_box.width,
                        "height": f.bounding_box.height,
                    }
                }
                for f in faces
            ]
        })
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({"error": "Detection failed"}), 500
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

---

## Client Example

### Python Client

```python
import requests

def verify_faces(image1_path, image2_path, api_url="http://localhost:8000"):
    """Verify two faces using the API."""
    with open(image1_path, "rb") as f1, open(image2_path, "rb") as f2:
        response = requests.post(
            f"{api_url}/verify",
            files={
                "image1": f1,
                "image2": f2,
            }
        )
    
    return response.json()


# Usage
result = verify_faces("person1.jpg", "person2.jpg")
print(f"Verified: {result['verified']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## Next Steps

- [Security Guide](../guides/security.md) - Secure your API
- [Deployment Guide](../guides/deployment.md) - Production deployment
