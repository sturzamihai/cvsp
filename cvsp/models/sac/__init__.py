import torch
import numpy as np

from cvsp.models.sac.adapter import AdversarialPatchDetector
from cvsp.models.sac.model import PatchDetector


def _preprocessing(rgb):
    return [torch.tensor(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)]


def load_model(checkpoint_path, device="cpu"):
    model = AdversarialPatchDetector(checkpoint_path, device=device)

    return model, _preprocessing
