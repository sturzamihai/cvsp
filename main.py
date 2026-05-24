import torch
import numpy as np
from PIL import Image
from models.gend import GenD, GenDConfig
from huggingface_hub import hf_hub_download
from lightning.fabric import Fabric
from models.lnclip_df.model import DeepfakeDetectionModel
from models.lnclip_df.config import Config
from transformers import AutoImageProcessor, AutoModelForImageClassification

paths = [
    "./data/Celeb-DF-v2/Celeb-synthesis/frames/id0_id1_0000/000.png",
    "./data/Celeb-DF-v2/Celeb-synthesis/frames/id0_id1_0000/045.png",
    "./data/Celeb-DF-v2/Celeb-synthesis/frames/id0_id1_0000/030.png",
    "./data/Celeb-DF-v2/Celeb-synthesis/frames/id0_id1_0000/015.png",
    "./data/Celeb-DF-v2/YouTube-real/frames/00000/000.png",
    "./data/Celeb-DF-v2/YouTube-real/frames/00000/014.png",
    "./data/Celeb-DF-v2/YouTube-real/frames/00000/028.png",
    "./data/Celeb-DF-v2/YouTube-real/frames/00000/043.png",
    "./data/Celeb-DF-v2/Celeb-real/frames/id0_0000/045.png",
    "./data/Celeb-DF-v2/Celeb-real/frames/id0_0000/030.png",
    "./data/Celeb-DF-v2/Celeb-real/frames/id0_0000/015.png",
    "./data/Celeb-DF-v2/Celeb-real/frames/id0_0000/000.png",
    "./data/IMG_5649.png",
]


def test_vit(images):
    processor = AutoImageProcessor.from_pretrained(
        "buildborderless/CommunityForensics-DeepfakeDet-ViT",
        size={"height": 384, "width": 384},
        do_center_crop=True,
    )
    model = AutoModelForImageClassification.from_pretrained(
        "buildborderless/CommunityForensics-DeepfakeDet-ViT"
    )

    tensors = processor(images, return_tensors="pt")

    with torch.no_grad():
        logits = model(**tensors).logits

    return logits


def test_lnclip(images):
    ckpt = torch.load("./weights/model.ckpt", map_location="cpu")
    run_name = ckpt["hyper_parameters"]["run_name"]

    print(run_name)

    model = DeepfakeDetectionModel(Config(**ckpt["hyper_parameters"]))
    model.eval()
    model.load_state_dict(ckpt["state_dict"])

    preprocessing = model.get_preprocessing()

    batch_images = torch.stack([preprocessing(image) for image in images])

    precision = ckpt["hyper_parameters"]["precision"]
    fabric = Fabric(precision=precision)
    fabric.launch()

    model = fabric.setup_module(model)

    with torch.no_grad():
        batch_images = batch_images.to(fabric.device).to(model.dtype)
        output = model(batch_images)

    return output.logits_labels.softmax(dim=1).cpu().numpy()


def test_gend(images):
    config = GenDConfig.from_pretrained("yermandy/GenD_CLIP_L_14")
    model = GenD(config)

    try:
        weights_path = hf_hub_download("yermandy/GenD_CLIP_L_14", "model.safetensors")
        from safetensors.torch import load_file

        state_dict = load_file(weights_path)
    except Exception:
        weights_path = hf_hub_download("yermandy/GenD_CLIP_L_14", "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    tensors = torch.stack(
        [model.feature_extractor.preprocess(image) for image in images]
    )
    logits = model(tensors)

    return logits.softmax(dim=-1)


class_mapping = {0: "REAL", 1: "FAKE"}

if __name__ == "__main__":
    images = [Image.open(path) for path in paths]

    vit_output = test_vit(images)
    lnclip_output = test_lnclip(images)
    gend_output = test_gend(images)

    for idx, image in enumerate(images):
        print(
            f"\nImage: {image.filename}\n",
            f"\tViT: {vit_output[idx]}, {class_mapping[np.argmax(vit_output[idx].numpy())]} \n",
            f"\tLNCLIP: {lnclip_output[idx]}, {class_mapping[np.argmax(lnclip_output[idx])]}\n",
            f"\tGenD: {gend_output[idx]}, {class_mapping[np.argmax(gend_output[idx].detach().numpy())]}\n",
        )
