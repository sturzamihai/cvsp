import torch
import numpy as np

from cvsp.models.sac.adapter import AdversarialPatchDetector
from cvsp.models.sac.model import PatchDetector


def _preprocessing(rgb):
    image = np.stack([rgb], axis=0).astype(np.float32) / 255.0

    image = image.transpose(0, 3, 1, 2)

    return torch.tensor(image)


def load_model(checkpoint_path, device="cpu"):
    model = AdversarialPatchDetector(checkpoint_path, device=device)

    return model, _preprocessing
