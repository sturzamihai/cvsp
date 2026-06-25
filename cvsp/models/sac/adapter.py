import torch
import numpy as np
import cv2

from cvsp.models.sac.model import PatchDetector


class AdversarialPatchDetector:
    def __init__(
        self,
        checkpoint_path,
        base_filter=64,
        open_kernel_size=15,
        close_kernel_size=21,
        min_bbox_area_ratio=0.02,
        min_fill_ratio=0.35,
        device="cpu",
    ):
        ckpt = torch.load(checkpoint_path, map_location=device)

        self.model = PatchDetector(
            3,
            1,
            n_patch=1,
            base_filter=base_filter,
            device=device,
        )
        self.model.unet.load_state_dict(ckpt)
        self.model.unet.to(device)
        self.model.unet.eval()

        self.open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
        self.close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)

        self.min_bbox_area_ratio = min_bbox_area_ratio
        self.min_fill_ratio = min_fill_ratio

    def __call__(self, inputs):
        with torch.no_grad():
            _, _, raw_masks = self.model(inputs, bpda=True, shape_completion=False)

        mask_np = (raw_masks[0][0, 0].cpu().detach().numpy() * 255).astype(np.uint8)

        mask_clean = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, self.open_kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, self.close_kernel)

        h_img, w_img = mask_clean.shape
        image_area = h_img * w_img
        min_bbox_area = self.min_bbox_area_ratio * image_area

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_clean)
        detections = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_bbox_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            bbox_area = w * h
            fill_ratio = area / bbox_area
            if fill_ratio < self.min_fill_ratio:
                continue

            relative_area = bbox_area / image_area
            confidence = fill_ratio * min(relative_area / self.min_bbox_area_ratio, 1)

            detections.append({"bbox": (x, y, w, h), "confidence": confidence})

        return detections
