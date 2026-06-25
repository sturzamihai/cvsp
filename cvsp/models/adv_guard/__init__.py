import torch
from torchvision import transforms

from cvsp.models.adv_guard.model import AdvGuard


def _preprocessing(rgb):
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    return transform(rgb)


def load_model(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = AdvGuard()
    model.to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, _preprocessing
