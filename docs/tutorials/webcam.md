# Webcam Verification Tutorial

Real-time face verification using webcam.

---

## Prerequisites

- FaceVerify installed
- OpenCV: `pip install opencv-python`
- Webcam connected

---

## Basic Webcam Capture

```python
"""Real-time face detection with webcam."""

import cv2
from faceverify import FaceVerifier

# Initialize
verifier = FaceVerifier()

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    try:
        faces = verifier.detect_faces(rgb_frame, return_all=True)
        if not isinstance(faces, list):
            faces = [faces] if faces else []
        
        # Draw boxes
        for face in faces:
            bbox = face.bounding_box
            cv2.rectangle(
                frame,
                (bbox.x, bbox.y),
                (bbox.x + bbox.width, bbox.y + bbox.height),
                (0, 255, 0),
                2
            )
    except Exception:
        pass
    
    # Display
    cv2.imshow("Face Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Verification Against Reference

```python
"""Verify webcam face against reference image."""

import cv2
import numpy as np
from faceverify import FaceVerifier

# Initialize
verifier = FaceVerifier()

# Load reference image and extract embedding
reference_path = "reference_person.jpg"
print(f"Loading reference: {reference_path}")
reference_embedding = verifier.extract_embedding(reference_path).embedding

# Open webcam
cap = cv2.VideoCapture(0)

print("Controls:")
print("  'v' - Verify current frame")
print("  'q' - Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Display
    display = frame.copy()
    cv2.putText(
        display,
        "Press 'v' to verify, 'q' to quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    cv2.imshow("Webcam Verification", display)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    elif key == ord('v'):
        # Extract embedding from current frame
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_embedding = verifier.extract_embedding(rgb_frame).embedding
            
            # Compare
            similarity, distance = verifier.compare_embeddings(
                reference_embedding,
                current_embedding
            )
            
            # Check threshold
            verified = similarity >= verifier.threshold
            
            # Show result
            color = (0, 255, 0) if verified else (0, 0, 255)
            status = "VERIFIED" if verified else "NOT VERIFIED"
            
            result_frame = frame.copy()
            cv2.putText(
                result_frame,
                f"{status} ({similarity:.2%})",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2
            )
            
            cv2.imshow("Webcam Verification", result_frame)
            cv2.waitKey(2000)  # Show result for 2 seconds
            
        except Exception as e:
            print(f"Verification failed: {e}")

cap.release()
cv2.destroyAllWindows()
```

---

## Continuous Verification

```python
"""Continuous real-time verification."""

import cv2
import time
from faceverify import FaceVerifier

verifier = FaceVerifier()

# Load reference
reference_embedding = verifier.extract_embedding("reference.jpg").embedding

cap = cv2.VideoCapture(0)

last_verify_time = 0
verify_interval = 1.0  # Verify every 1 second
last_result = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    current_time = time.time()
    
    # Verify at intervals
    if current_time - last_verify_time >= verify_interval:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_embedding = verifier.extract_embedding(rgb_frame).embedding
            
            similarity, _ = verifier.compare_embeddings(
                reference_embedding,
                current_embedding
            )
            
            last_result = {
                "verified": similarity >= verifier.threshold,
                "similarity": similarity,
            }
            last_verify_time = current_time
            
        except Exception:
            last_result = None
    
    # Draw result
    if last_result:
        color = (0, 255, 0) if last_result["verified"] else (0, 0, 255)
        status = "VERIFIED" if last_result["verified"] else "NOT VERIFIED"
        
        cv2.putText(
            frame,
            f"{status} ({last_result['similarity']:.2%})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )
    
    cv2.imshow("Continuous Verification", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Performance Tips

> [!TIP]
> For smoother webcam experience:
> - Use `detector_backend="opencv"` (fastest)
> - Reduce frame size before processing
> - Verify at intervals, not every frame
> - Cache reference embeddings

---

## Next Steps

- [Performance Guide](../guides/performance.md) - Optimize speed
- [REST API Tutorial](rest-api.md) - Build web service
