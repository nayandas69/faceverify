# Webcam Verification Tutorial

Real-time face verification using webcam.

---

## Prerequisites

- FaceVerify installed
- OpenCV: `pip install opencv-python`
- Webcam connected

---

## Basic Webcam Capture

### Capture Reference Image

```python
"""Capture a reference face image from webcam."""

import cv2

def capture_reference():
    """Capture a single frame as reference."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return None
    
    print("Press SPACE to capture, ESC to cancel")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display
        cv2.imshow("Capture Reference", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            cv2.imwrite("reference.jpg", frame)
            print("Reference saved to reference.jpg")
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_reference()
```

---

## Real-Time Verification

### Continuous Verification Against Reference

```python
"""Real-time face verification against a reference image."""

import cv2
from faceverify import FaceVerifier
from faceverify.config import VerifierConfig

# Initialize
config = VerifierConfig(
    detector_backend="opencv",  # Fast for real-time
    threshold=0.65,
)
verifier = FaceVerifier(config)

# Reference image path
REFERENCE_IMAGE = "reference.jpg"


def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    print("Real-time Face Verification")
    print("Controls:")
    print("  SPACE - Verify current frame")
    print("  R     - Capture new reference")
    print("  ESC   - Exit")
    
    reference_path = REFERENCE_IMAGE
    last_result = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display frame
        display = frame.copy()
        
        # Show last result
        if last_result is not None:
            if last_result.verified:
                color = (0, 255, 0)  # Green
                text = f"VERIFIED ({last_result.confidence:.1%})"
            else:
                color = (0, 0, 255)  # Red
                text = f"NOT VERIFIED ({last_result.confidence:.1%})"
            
            cv2.putText(display, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Face Verification", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC - Exit
            break
        
        elif key == 32:  # SPACE - Verify
            # Save current frame temporarily
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            try:
                last_result = verifier.verify(reference_path, temp_path)
                print(f"Verified: {last_result.verified}, "
                      f"Confidence: {last_result.confidence:.2%}")
            except Exception as e:
                print(f"Verification error: {e}")
                last_result = None
        
        elif key == ord('r'):  # R - New reference
            cv2.imwrite(reference_path, frame)
            print(f"New reference saved to {reference_path}")
            last_result = None
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

---

## Automatic Continuous Verification

### Verify Every N Frames

```python
"""Automatic continuous face verification."""

import cv2
import time
from faceverify import FaceVerifier
from faceverify.config import VerifierConfig

config = VerifierConfig(
    detector_backend="opencv",
    threshold=0.65,
)
verifier = FaceVerifier(config)

REFERENCE_IMAGE = "reference.jpg"
VERIFY_INTERVAL = 2.0  # Seconds between verifications


def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    print("Automatic Face Verification")
    print(f"Verifying every {VERIFY_INTERVAL} seconds")
    print("Press ESC to exit")
    
    last_verify_time = 0
    last_result = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        
        # Auto-verify at interval
        if current_time - last_verify_time >= VERIFY_INTERVAL:
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            try:
                last_result = verifier.verify(REFERENCE_IMAGE, temp_path)
                status = "MATCH" if last_result.verified else "NO MATCH"
                print(f"[{time.strftime('%H:%M:%S')}] {status} "
                      f"(confidence: {last_result.confidence:.2%})")
            except Exception as e:
                print(f"Error: {e}")
                last_result = None
            
            last_verify_time = current_time
        
        # Display
        display = frame.copy()
        
        if last_result is not None:
            color = (0, 255, 0) if last_result.verified else (0, 0, 255)
            text = f"{'VERIFIED' if last_result.verified else 'NOT VERIFIED'}"
            cv2.putText(display, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(display, f"Confidence: {last_result.confidence:.1%}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("Auto Verification", display)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

---

## Performance Tips

> [!TIP]
> For smooth real-time performance:
> - Use `opencv` detector (fastest)
> - Reduce frame resolution if needed
> - Verify every few frames, not every frame

### Reduce Resolution

```python
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

---

## Troubleshooting

### Webcam Not Opening

```python
# Try different camera indices
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        break
```

### Slow Verification

> [!NOTE]
> First verification is slow (model loading). Subsequent verifications are faster.

---

## Next Steps

- [REST API Tutorial](rest-api.md) - Build an API
- [Performance Guide](../guides/performance.md) - Optimization tips
