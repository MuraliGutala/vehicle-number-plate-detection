import cv2
from ultralytics import YOLO
import easyocr

# Load YOLO model (must have license_plate_model.pt in your folder)
model = YOLO("license_plate_detector.pt")

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Open camera
cap = cv2.VideoCapture(0)

print("Press SPACE to capture a photo, ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Camera - Press SPACE to Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == 32:  # SPACE to capture
        captured = frame.copy()
        cv2.imshow("Captured Image", captured)

        # Run YOLO detection
        results = model(captured, imgsz=640)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = captured[y1:y2, x1:x2]

                if plate_crop.size == 0:
                    continue

                # Preprocess for OCR
                plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)
                _, plate_thresh = cv2.threshold(plate_gray, 150, 255, cv2.THRESH_BINARY)

                # OCR
                ocr_result = reader.readtext(plate_thresh, detail=0)
                text = " ".join(ocr_result)

                # Draw result
                cv2.rectangle(captured, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(captured, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                print("Detected Plate:", text)

        cv2.imshow("Detection Result", captured)

cap.release()
cv2.destroyAllWindows()
