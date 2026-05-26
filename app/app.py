import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchattacks
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from cvsp.preprocessing import extract_aligned_face_dlib
from cvsp.models.lnclip_df import load_model as load_lnclip_model
from cvsp.models.minifas import FaceAntiSpoofing
from cvsp.models.flip import load_model as load_flip_model

import dlib

PREDICTOR_PATH = ROOT / "weights" / "shape_predictor_81_face_landmarks.dat"
LNCLIP_CKPT = ROOT / "weights" / "lnclip.ckpt"
FLIP_CKPT = ROOT / "weights" / "wmca_flip_mcl.pth.tar"
BATCH_SIZE = 48  # frames averaged per verdict


def _load_dlib():
    detector = dlib.get_frontal_face_detector()
    predictor = None
    if PREDICTOR_PATH.exists():
        predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
    else:
        print(
            f"[WARNING] Shape predictor not found at {PREDICTOR_PATH}. Alignment disabled."
        )
    return detector, predictor


_detector, _predictor = _load_dlib()
_lnclip, _lnclip_preprocess = load_lnclip_model(LNCLIP_CKPT)
_flip, _flip_preprocess = load_flip_model(FLIP_CKPT)
_antispoof = FaceAntiSpoofing()


class _LogitsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits_labels


_pgd_attack = (
    torchattacks.PGD(_LogitsWrapper(_lnclip), eps=8 / 255, alpha=2 / 255, steps=25)
    if _lnclip is not None
    else None
)


def _get_largest_dlib_face(rgb: np.ndarray) -> dlib.rectangle | None:
    """Returns the largest dlib face rect in full-res coordinates."""
    faces = list(_detector(rgb, 1))
    if not faces:
        return None
    return max(faces, key=lambda f: f.width() * f.height())


def _get_aligned_pil(
    rgb: np.ndarray, scale: float = 1.3, use_eye_centers: bool = False
) -> Image.Image | None:
    """
    Run the full DFB alignment pipeline on one RGB frame.
    Alignment runs on the original full-resolution frame so the 256×256 crop
    matches the quality of the offline training preprocessing.
    Returns a PIL Image (RGB) or None if no face found.
    """
    if _predictor is None:
        return None
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cropped_bgr, _, _ = extract_aligned_face_dlib(
        _detector, _predictor, bgr, scale=scale, use_eye_centers=use_eye_centers
    )
    if cropped_bgr is None:
        return None
    return Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))


def _run_lnclip(
    pil_images: list[Image.Image], apply_attack: bool = False
) -> torch.Tensor:
    """Returns (N, 2) softmax probabilities [P(REAL), P(FAKE)]."""
    tensors = torch.stack([_lnclip_preprocess(img) for img in pil_images])
    if apply_attack and _pgd_attack is not None:
        # Label 1 = FAKE; PGD maximises loss for this label, pushing predictions toward REAL
        labels = torch.ones(len(tensors), dtype=torch.long)
        tensors = _pgd_attack(tensors, labels)
    with torch.no_grad():
        return _lnclip(tensors).logits_labels.softmax(dim=1)


def _run_flip(pil_images: list[Image.Image]) -> torch.Tensor:
    """Returns (N, 1) softmax probabilities [0 = fake, 1 = true]."""
    tensors = torch.stack([_flip_preprocess(img) for img in pil_images])

    with torch.no_grad():
        cls_out, _ = _flip.forward_eval(tensors, True)
    print(
        f"[FLIP] raw logits mean: spoof={cls_out[:,0].mean():.3f} real={cls_out[:,1].mean():.3f}"
    )
    return F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]


def _to_label(probs: list[float]) -> dict:
    """Format [p_real, p_fake] for gr.Label."""
    return {"REAL": float(probs[0]), "FAKE": float(probs[1])}


def process_video(video_path: str, apply_attack: bool = False, progress=gr.Progress()):
    if video_path is None:
        return None, None, None, [], []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, [], []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # CAP_PROP_FRAME_COUNT is unreliable for webcam recordings — the container
    # header isn't written until the file is closed, so it often reads 0 or 1.
    # Use grab() (no decode) to count actual frames, then reopen.
    if total <= 1:
        count = 0
        while cap.grab():
            count += 1
        total = count
        cap.release()
        cap = cv2.VideoCapture(video_path)

    # Sample BATCH_SIZE indices uniformly
    n_samples = min(BATCH_SIZE, max(total, 1))
    sample_idxs = set(np.linspace(0, total - 1, n_samples, dtype=int).tolist())

    aligned_pils: list[Image.Image] = []
    flip_pils: list[Image.Image] = []
    spoof_results: list[tuple[bool, float]] = []
    frames_processed = 0

    progress(0, desc="Sampling frames…")
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frames_processed in sample_idxs:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dlib_face = _get_largest_dlib_face(rgb)
            if dlib_face is not None:
                spoof = _antispoof(rgb, dlib_face)
                if spoof is not None:
                    spoof_results.append(spoof)
            aligned = _get_aligned_pil(rgb)
            if aligned is not None:
                aligned_pils.append(aligned)
            flip_aligned = _get_aligned_pil(rgb, scale=1.0, use_eye_centers=True)
            if flip_aligned is not None:
                flip_pils.append(flip_aligned)

        frames_processed += 1
        if total > 0:
            progress(frames_processed / total * 0.80, desc="Sampling frames…")

    cap.release()

    lnclip_label_out = None
    flip_label_out = None
    minifas_label_out = None

    if spoof_results:
        avg_live = sum(score for _, score in spoof_results) / len(spoof_results)
        minifas_label_out = {"LIVE": avg_live, "SPOOF": 1.0 - avg_live}

    if aligned_pils and _lnclip is not None:
        progress(0.85, desc="Running LNCLIP-DF...")
        avg = _run_lnclip(aligned_pils, apply_attack=apply_attack).mean(dim=0).tolist()
        lnclip_label_out = _to_label(avg)

    if flip_pils and _flip is not None:
        progress(0.95, desc="Running FLIP...")
        avg = _run_flip(flip_pils).mean(axis=0).tolist()
        flip_label_out = {"SPOOFED": 1 - avg, "LIVE": avg}

    progress(1.0)

    max_preview = 8
    lnclip_preview = aligned_pils[::max(1, len(aligned_pils) // max_preview)][:max_preview]
    flip_preview = flip_pils[::max(1, len(flip_pils) // max_preview)][:max_preview]

    return lnclip_label_out, flip_label_out, minifas_label_out, lnclip_preview, flip_preview


_status = (
    f"dlib: **{'✓' if PREDICTOR_PATH.exists() else '✗ not found'}**  |  "
    f"LNCLIP-DF: **{'✓' if _lnclip is not None else '✗ not loaded'}**"
)

with gr.Blocks(title="DeepFakeBench + LNCLIP-DF") as demo:
    gr.Markdown(
        "# DeepFakeBench Preprocessing + LNCLIP-DF Detection\n"
        f"{_status}\n\n"
        f"Record from your webcam or upload a video. "
        f"The app samples **{BATCH_SIZE} frames uniformly**, aligns faces with the "
        f"DFB similarity-transform pipeline, and averages LNCLIP-DF softmax "
        f"probabilities for a video-level deepfake verdict."
    )

    video_in = gr.Video(
        sources=["webcam", "upload"],
        label="Input — record or upload",
    )

    with gr.Row():
        analyze_btn = gr.Button("Analyze", variant="primary")
        attack_toggle = gr.Checkbox(
            label="Apply adversarial attack",
            value=False,
            info="Applies PGD perturbation to sampled frames to evade the deepfake detector. Enable for deepfake videos.",
        )

    with gr.Row():
        verdict_label = gr.Label(num_top_classes=2, label="Deepfake (LNCLIP-DF)")
        flip_label = gr.Label(num_top_classes=2, label="Presentation Attack (FLIP)")
        minifas_label = gr.Label(num_top_classes=2, label="Presentation Attack (MiniFAS)")

    with gr.Row():
        lnclip_gallery = gr.Gallery(
            label="LNCLIP-DF aligned crops", columns=8, height=160, object_fit="cover"
        )
        flip_gallery = gr.Gallery(
            label="FLIP aligned crops", columns=8, height=160, object_fit="cover"
        )

    analyze_btn.click(
        fn=process_video,
        inputs=[video_in, attack_toggle],
        outputs=[verdict_label, flip_label, minifas_label, lnclip_gallery, flip_gallery],
    )


if __name__ == "__main__":
    demo.launch()
