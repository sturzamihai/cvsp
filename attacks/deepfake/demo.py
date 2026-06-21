from pathlib import Path

import cv2

from attacks.deepfake.attack import FaceSwap

ROOT = Path(__file__).parent.parent.parent
INSWAPPER_PATH = ROOT / "weights" / "inswapper_128.onnx"
TARGET_PATH = Path(__file__).parent / "target.jpg"

swap = FaceSwap(INSWAPPER_PATH)

target = cv2.imread(str(TARGET_PATH))
target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
if not swap.prepare(target_rgb):
    raise RuntimeError("No face found in target image or insightface unavailable.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = swap.apply(frame)
    cv2.imshow("Deepfake", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
