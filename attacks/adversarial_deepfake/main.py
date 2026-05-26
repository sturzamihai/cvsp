import torch
import torch.nn as nn
from PIL import Image
import torchattacks
from cvsp.models.lnclip_df import load_model as make_deepfake_model


class LogitsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits_labels


device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocessing = make_deepfake_model("./weights/lnclip.ckpt", device=device)
attack = torchattacks.PGD(LogitsWrapper(model), eps=8 / 255, alpha=2 / 255, steps=25)

image_path = "./data/Celeb-DF-v2/Celeb-synthesis/frames/id0_id1_0000/045.png"
image = preprocessing(Image.open(image_path))

images = torch.stack([image]).to(device)
labels = torch.tensor([1], dtype=torch.long).to(device)
adv_image = attack(images, labels)

output = model(adv_image)
results = output.logits_labels.softmax(dim=1).cpu()
print(results)
