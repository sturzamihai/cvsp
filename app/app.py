import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torchattacks
from PIL import Image

try:
    import insightface
    from insightface.app import FaceAnalysis as InsightFaceAnalysis

    _insightface_available = True
except ImportError:
    _insightface_available = False

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from cvsp.preprocessing import extract_aligned_face_dlib
from cvsp.models.lnclip_df import load_model as load_lnclip_model
from cvsp.models.minifas import FaceAntiSpoofing
from cvsp.models.sac import load_model as load_sac_model

import dlib

PREDICTOR_PATH = ROOT / "weights" / "shape_predictor_81_face_landmarks.dat"
LNCLIP_CKPT = ROOT / "weights" / "lnclip.ckpt"
SAC_CKPT = ROOT / "weights" / "apricot_mask.pth"
INSWAPPER_PATH = ROOT / "weights" / "inswapper_128.onnx"
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


_device = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

_detector, _predictor = _load_dlib()
_lnclip, _lnclip_preprocess = load_lnclip_model(LNCLIP_CKPT)
_antispoof = FaceAntiSpoofing()
_sac, _sac_preprocess = load_sac_model(str(SAC_CKPT), device=_device)

_fa_app = None
_swapper = None


def _get_deepfake_models():
    global _fa_app, _swapper
    if not _insightface_available:
        return None, None
    if _fa_app is None:
        _fa_app = InsightFaceAnalysis()
        _fa_app.prepare(ctx_id=0)
    if _swapper is None and INSWAPPER_PATH.exists():
        _swapper = insightface.model_zoo.get_model(str(INSWAPPER_PATH))
    return _fa_app, _swapper


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
    faces = list(_detector(rgb, 1))
    if not faces:
        return None
    return max(faces, key=lambda f: f.width() * f.height())


def _get_aligned_pil(
    rgb: np.ndarray, scale: float = 1.3, use_eye_centers: bool = False
) -> Image.Image | None:
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


def _to_label(probs: list[float]) -> dict:
    return {"REAL": float(probs[0]), "FAKE": float(probs[1])}


def process_video(
    video_path: str,
    apply_attack: bool = False,
    apply_deepfake: bool = False,
    target_image: np.ndarray | None = None,
    progress=gr.Progress(),
):
    if video_path is None:
        return None, None, []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, []

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

    # Prepare deepfake source face once before the frame loop
    source_face = None
    fa_app = None
    swapper = None
    if apply_deepfake and target_image is not None:
        fa_app, swapper = _get_deepfake_models()
        if fa_app is not None and swapper is not None:
            # Gradio returns RGB numpy arrays; insightface expects BGR
            target_bgr = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
            faces = fa_app.get(target_bgr)
            if faces:
                source_face = faces[0]

    # Sample BATCH_SIZE indices uniformly
    n_samples = min(BATCH_SIZE, max(total, 1))
    sample_idxs = set(np.linspace(0, total - 1, n_samples, dtype=int).tolist())

    aligned_pils: list[Image.Image] = []
    spoof_results: list[tuple[bool, float]] = []
    sac_scores: list[float] = []
    frames_processed = 0

    progress(0, desc="Sampling frames…")
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frames_processed in sample_idxs:
            if source_face is not None:
                frame_faces = fa_app.get(bgr)
                if frame_faces:
                    bgr = swapper.get(bgr, frame_faces[0], source_face, paste_back=True)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if _sac is not None:
                detections = _sac(_sac_preprocess(rgb))
                sac_scores.append(
                    max(d["confidence"] for d in detections) if detections else 0.0
                )

            dlib_face = _get_largest_dlib_face(rgb)
            if dlib_face is not None:
                spoof = _antispoof(rgb, dlib_face)
                if spoof is not None:
                    spoof_results.append(spoof)
            aligned = _get_aligned_pil(rgb)
            if aligned is not None:
                aligned_pils.append(aligned)

        frames_processed += 1
        if total > 0:
            progress(frames_processed / total * 0.80, desc="Sampling frames…")

    cap.release()

    lnclip_label_out = None
    minifas_label_out = None
    sac_label_out = None

    if spoof_results:
        avg_live = sum(score for _, score in spoof_results) / len(spoof_results)
        minifas_label_out = {"LIVE": avg_live, "SPOOF": 1.0 - avg_live}

    if sac_scores:
        patch_conf = sum(sac_scores) / len(sac_scores)
        sac_label_out = {"PATCH": patch_conf, "CLEAN": 1.0 - patch_conf}

    if aligned_pils and _lnclip is not None:
        progress(0.85, desc="Running LNCLIP-DF...")
        avg = _run_lnclip(aligned_pils, apply_attack=apply_attack).mean(dim=0).tolist()
        lnclip_label_out = _to_label(avg)

    progress(1.0)

    max_preview = 8
    lnclip_preview = aligned_pils[:: max(1, len(aligned_pils) // max_preview)][
        :max_preview
    ]

    return lnclip_label_out, minifas_label_out, sac_label_out, lnclip_preview


with gr.Blocks(title="DeepFakeBench + LNCLIP-DF") as demo:
    gr.Markdown("# Computer Vision Spoofing Prevention System with AI\n")

    with gr.Row():
        video_in = gr.Video(
            sources=["webcam", "upload"],
            label="Input — record or upload",
        )
        with gr.Column():
            attack_toggle = gr.Checkbox(
                label="Apply adversarial attack",
                value=False,
                info="Applies PGD perturbation to sampled frames to evade the deepfake detector.",
            )
            deepfake_toggle = gr.Checkbox(
                label="Apply deepfake",
                value=False,
                info="Swaps faces in the video with the target face using inswapper.",
            )
            target_image = gr.Image(
                label="Target face",
                type="numpy",
                visible=False,
            )

    analyze_btn = gr.Button("Analyze", variant="primary")

    with gr.Row():
        verdict_label = gr.Label(num_top_classes=2, label="Deepfake (LNCLIP-DF)")
        minifas_label = gr.Label(
            num_top_classes=2, label="Presentation Attack (MiniFAS)"
        )
        sac_label = gr.Label(num_top_classes=2, label="Patch Attack (SAC)")

    lnclip_gallery = gr.Gallery(
        label="LNCLIP-DF aligned crops", columns=8, height=160, object_fit="cover"
    )

    deepfake_toggle.change(
        fn=lambda enabled: gr.update(visible=enabled),
        inputs=[deepfake_toggle],
        outputs=[target_image],
    )

    analyze_btn.click(
        fn=process_video,
        inputs=[video_in, attack_toggle, deepfake_toggle, target_image],
        outputs=[verdict_label, minifas_label, sac_label, lnclip_gallery],
    )


if __name__ == "__main__":
    demo.launch()
