import torchattacks
from cvsp.models.lnclip_df import make_model as make_deepfake_model

model = make_deepfake_model("../weights/lnclip.ckpt")
attack = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=25)
