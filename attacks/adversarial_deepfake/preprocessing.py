import torch
import torch.nn as nn

from torchvision import transforms

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class NormalizeWrapper(nn.Module):
    def __init__(self, model: nn.Module, mean=_CLIP_MEAN, std=_CLIP_STD) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x_norm = (x.float() - mean) / std
        model_dtype = next(self.model.parameters()).dtype
        return self.model(x_norm.to(model_dtype)).logits_labels.float()


def build_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
