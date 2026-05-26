import cv2
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis()
app.prepare(ctx_id=0)

source_img = cv2.imread("./attacks/deepfake/target.jpg")
source_face = app.get(source_img)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    target_face = app.get(frame)
    swapper = insightface.model_zoo.get_model("./weights/inswapper_128.onnx")
    result = swapper.get(frame, target_face[0], source_face[0], paste_back=True)

    cv2.imshow("Deepfake", result)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
