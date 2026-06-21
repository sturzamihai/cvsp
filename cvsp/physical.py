import torch
import cv2

from cvsp.models.minifas import FaceAntiSpoofing
from cvsp.models.sac import load_model as load_sac_model


class PhysicalDefense:
    def __init__(self, sac_checkpoint_path, device="cpu"):
        self.antispoofing = FaceAntiSpoofing()
        self.sac, self.sac_preprocessing = load_sac_model(
            sac_checkpoint_path, device=device
        )

    def _resize(self, img, max_dim=640):
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        if scale >= 1.0:
            return img
        return cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    def __call__(self, images, input_is_bgr: bool = False):
        rgb_images = images
        if input_is_bgr:
            rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

        rgb_images = [self._resize(img) for img in rgb_images]

        sac_scores = []
        spoof_scores = []
        for image in rgb_images:
            with torch.no_grad():
                sac_detection = self.sac(self.sac_preprocessing(image))
            spoof_detection = self.antispoofing(image)

            sac_scores.append(
                max(d["confidence"] for d in sac_detection) if sac_detection else 0.0
            )
            spoof_scores.append(spoof_detection)

        avg_live = sum(score for _, score in spoof_scores) / len(spoof_scores)
        avg_patch = sum(sac_scores) / len(sac_scores)
        return {
            "spoofed": [avg_live, 1 - avg_live],
            "patch": [1 - avg_patch, avg_patch],
        }
