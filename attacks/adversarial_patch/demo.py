import cv2
from cvsp.models.sac import load_model

model, preprocessing = load_model("./weights/apricot_mask.pth")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (854, 480))
    rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    detections = model(preprocessing(rgb))

    display = resized_frame.copy()
    if detections:
        best = max(detections, key=lambda d: d["confidence"])
        bx, by, bw, bh = best["bbox"]
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
        label = f"ATTACKED  {best['confidence']:.2f}"
        color = (0, 0, 255)
    else:
        label = "CLEAN"
        color = (0, 255, 0)

    cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("SAC", display)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
