from ultralytics import YOLO
import cv2

# Load YOLO11s model
model = YOLO("yolo11s.pt")

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Run object detection
    results = model(
        frame,
        imgsz=640,
        conf=0.4,
        verbose=False
    )

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO11 Object Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()