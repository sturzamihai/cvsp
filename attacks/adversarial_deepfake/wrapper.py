import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from attacks.adversarial_deepfake.attack import FGSMAttack
from attacks.adversarial_deepfake.preprocessing import (
    NormalizeWrapper,
    build_preprocess,
)


class AdversarialFGSM:
    EPS = 8 / 255

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self._preprocess = build_preprocess()
        self._to_pil = transforms.ToPILImage()
        self._device = device
        wrapped = NormalizeWrapper(model).to(device)
        self._fgsm = FGSMAttack(wrapped, eps=self.EPS)

    def apply(self, pil_images: list[Image.Image]) -> list[Image.Image]:
        if not pil_images:
            return []
        tensors = torch.stack([self._preprocess(img) for img in pil_images]).to(
            self._device
        )
        # Label 1 = FAKE; FGSM pushes predictions toward REAL (label 0)
        labels = torch.ones(len(pil_images), dtype=torch.long, device=self._device)
        results = self._fgsm.apply(tensors, labels)
        return [self._to_pil(t.cpu().clamp(0, 1)) for _, t in results]
