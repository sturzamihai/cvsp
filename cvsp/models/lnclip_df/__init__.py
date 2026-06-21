import torch
import torch.nn as nn

from cvsp.models.lnclip_df.model import DeepfakeDetectionModel
from cvsp.models.lnclip_df.config import Config

_PRECISION_TO_DTYPE = {
    "bf16": torch.bfloat16,
    "bf16-mixed": torch.bfloat16,
    "16": torch.float16,
    "16-mixed": torch.float16,
}


class LogitsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits_labels


def load_model(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = DeepfakeDetectionModel(Config(**ckpt["hyper_parameters"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    precision = ckpt["hyper_parameters"].get("precision", "32")
    dtype = _PRECISION_TO_DTYPE.get(precision)
    if dtype is not None:
        model = model.to(dtype)

    model = model.to(device)
    preprocessing = model.get_preprocessing()

    return model, preprocessing
