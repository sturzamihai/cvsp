import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms


def tensor_to_uint8(t: torch.Tensor) -> torch.Tensor:
    return (t.float() * 255).round().clamp(0, 255).to(torch.uint8)


def save_png(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor_to_uint8(tensor).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(str(path), format="PNG")


def load_png_as_tensor(path: Path) -> torch.Tensor:
    return transforms.ToTensor()(Image.open(path).convert("RGB"))
