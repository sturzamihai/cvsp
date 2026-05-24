import torch
from lightning.fabric import Fabric

from model import DeepfakeDetectionModel
from config import Config


def make_model(
    checkpoint_path,
    device: torch.Device = "cpu",
):
    ckpt = torch.load(checkpoint_path, map_location=device)
    run_name = ckpt["hyper_parameters"]["run_name"]

    model = DeepfakeDetectionModel(Config(**ckpt["hyper_parameters"]))
    model.eval()
    model.load_state_dict(ckpt["state_dict"])

    preprocessing = model.get_preprocessing()

    precision = ckpt["hyper_parameters"]["precision"]
    fabric = Fabric(precision=precision)
    fabric.launch()

    model = fabric.setup_module(model)

    return model, preprocessing
