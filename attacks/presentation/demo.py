from pathlib import Path

import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        faces = DeepFace.extract_faces(rgb, anti_spoofing=True)
    except Exception:
        faces = []

    for face in faces:
        area = face.get("facial_area", {})
        x, y, w, h = (
            area.get("x", 0),
            area.get("y", 0),
            area.get("w", 0),
            area.get("h", 0),
        )
        is_real = face.get("is_real", True)

        color = (0, 255, 0) if is_real else (0, 0, 255)
        label = "clean" if is_real else "spoofed"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Presentation", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
