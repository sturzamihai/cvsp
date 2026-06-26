import numpy as np
import cv2
import torch

from cvsp.models.sac.model import PatchDetector

CHECKPOINT = "./weights/apricot_mask.pth"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

SAC_processor = PatchDetector(
    3,
    1,
    base_filter=64,
    square_sizes=[125, 100, 75, 50, 25],
    n_patch=1,
    device=device,
)
SAC_processor.unet.load_state_dict(torch.load(CHECKPOINT, map_location=device))
SAC_processor.unet.to(device)
SAC_processor.unet.eval()

cap = cv2.VideoCapture(0)

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (854, 480))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        x = torch.tensor(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

        _, _, raw_masks = SAC_processor([x], bpda=True, shape_completion=False)

        # raw_mask: values in {0, 1}, shape 1x1xHxW
        mask_np = raw_masks[0][0, 0].cpu().numpy()  # HxW, float32

        # red overlay where the network predicts a patch
        overlay = resized.copy()
        overlay[mask_np > 0.5] = (0, 0, 255)
        display = cv2.addWeighted(resized, 0.6, overlay, 0.4, 0)

        cv2.imshow("SAC Raw", display)

        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
