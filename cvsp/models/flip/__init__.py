import torch
import torchvision.transforms as T
from torch.nn import functional as F

from cvsp.models.flip.model import flip_mcl


def load_model(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = flip_mcl(in_dim=512, ssl_mlp_dim=4096, ssl_emb_dim=256).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preprocessing = T.Compose(
        [
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ]
    )

    return model, preprocessing


def process_output(model_output):
    prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]


if __name__ == "__main__":
    from PIL import Image

    model, preprocessing = load_model("./weights/surf_flip_mcl.pth.tar")

    image = Image.open(
        "/Users/mihaisturza/Projects/University/cvsp/data/Celeb-DF-v2/Celeb-real/frames/id0_0000/030.png"
    )
    input = preprocessing(image).unsqueeze(0)

    with torch.no_grad():
        cls_out, feature = model.forward_eval(input, True)
        prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]

        print(prob)
