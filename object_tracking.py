from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolo11s.pt")

tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)

    detections = []

    for result in results:
        boxes = result.boxes

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            conf = float(box.conf[0])

            cls = int(box.cls[0])

            label = model.names[cls]

            detections.append(
                ([x1, y1, x2 - x1, y2 - y1], conf, label)
            )

    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        x1, y1, x2, y2 = map(int, track.to_ltrb())

        label = track.get_det_class()

        text = f"{label} ID:{track_id}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("YOLO11 + Deep SORT", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()