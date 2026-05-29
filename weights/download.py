# run from inside the weights folder

import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

import gdown


def download_lnclip_df(repo_id="yermandy/deepfake-detection", filename="model.ckpt"):
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=".")
    dst = shutil.move(path, "lnclip.ckpt")

    return str(dst)


def download_dlib_predictor(
    url="https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/shape_predictor_81_face_landmarks.dat",
):
    dst = Path("shape_predictor_81_face_landmarks.dat")
    urllib.request.urlretrieve(url, dst)
    return str(dst)


def download_inswapper(
    url="https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx",
):
    dst = Path("inswapper_128.onnx")
    urllib.request.urlretrieve(url, dst)
    return str(dst)


def download_sac(gdrive_id="1yl_ZcrNFme_yP22UgynvG0fHtuy8XmcZ"):
    return gdown.download(id=gdrive_id, output="sac.pth")


if __name__ == "__main__":
    required_downloads = {
        "LNCLIP-DF": download_lnclip_df,
        "DLIB Shape Predictor": download_dlib_predictor,
        "InSwapper 128": download_inswapper,
        "SAC": download_sac,
    }

    for key, value in required_downloads.items():
        print(f"Downloading {key}...")
        path = value()
        print(f"{key} downloaded to {path}")
