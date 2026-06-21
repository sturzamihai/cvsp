from abc import ABC, abstractmethod
import io

import torch
import torch.nn as nn
from PIL import Image

import torchattacks
from torchvision import transforms

from attacks.utils.fs import tensor_to_uint8


class Attack(ABC):
    label: int

    @abstractmethod
    def apply(
        self, imgs: torch.Tensor, labels: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]: ...


class FGSMAttack(Attack):
    label = 1

    def __init__(self, model: nn.Module, eps: float) -> None:
        self._attack = torchattacks.FGSM(model, eps=eps)

    def apply(self, imgs, labels):
        adv = self._attack(imgs, labels)
        return [("fgsm", adv[i]) for i in range(len(imgs))]


class PGDAttack(Attack):
    label = 1

    def __init__(self, model: nn.Module, eps: float, alpha: float, steps: int) -> None:
        self._attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)

    def apply(self, imgs, labels):
        adv = self._attack(imgs, labels)
        return [("pgd", adv[i]) for i in range(len(imgs))]


class UniformNoiseAugmentation(Attack):
    label = 0
    _levels = [2 / 255, 4 / 255, 8 / 255, 16 / 255]

    def __init__(self) -> None:
        self._counter = 0

    def apply(self, imgs, _):
        results = []
        for img in imgs:
            eps = self._levels[self._counter % len(self._levels)]
            name = f"noise{int(round(eps * 255))}"
            results.append(
                (name, (img + torch.empty_like(img).uniform_(-eps, eps)).clamp(0, 1))
            )
            self._counter += 1
        return results


class JPEGAugmentation(Attack):
    label = 0
    _qualities = [95, 75, 50]

    def __init__(self) -> None:
        self._counter = 0

    def _jpeg_compress(self, x: torch.Tensor, quality: int) -> torch.Tensor:
        arr = tensor_to_uint8(x).permute(1, 2, 0).cpu().numpy()
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return transforms.ToTensor()(Image.open(buf).convert("RGB"))

    def apply(self, imgs, _):
        results = []
        for img in imgs:
            quality = self._qualities[self._counter % len(self._qualities)]
            results.append((f"jpeg{quality}", self._jpeg_compress(img, quality)))
            self._counter += 1
        return results
