import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torchattacks
from PIL import Image

from cvsp.physical import PhysicalDefense
from cvsp.digital import DigitalDefense

try:
    import insightface
    from insightface.app import FaceAnalysis as InsightFaceAnalysis

    _insightface_available = True
except ImportError:
    _insightface_available = False

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


PREDICTOR_PATH = ROOT / "weights" / "shape_predictor_81_face_landmarks.dat"
LNCLIP_CKPT = ROOT / "weights" / "lnclip.ckpt"
ADV_GUARD_CKPT = ROOT / "weights" / "adv_guard.pt"
SAC_CKPT = ROOT / "weights" / "apricot_mask.pth"
INSWAPPER_PATH = ROOT / "weights" / "inswapper_128.onnx"
BATCH_SIZE = 48

_device = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

_digital_defense = DigitalDefense(PREDICTOR_PATH, ADV_GUARD_CKPT, LNCLIP_CKPT)
_physical_defense = PhysicalDefense(SAC_CKPT, _device)

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

    if total <= 1:
        count = 0
        while cap.grab():
            count += 1
        total = count
        cap.release()
        cap = cv2.VideoCapture(video_path)

    source_face = None
    fa_app = None
    swapper = None
    if apply_deepfake and target_image is not None:
        fa_app, swapper = _get_deepfake_models()
        if fa_app is not None and swapper is not None:
            target_bgr = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
            faces = fa_app.get(target_bgr)
            if faces:
                source_face = faces[0]

    n_samples = min(BATCH_SIZE, max(total, 1))
    sample_idxs = set(np.linspace(0, total - 1, n_samples, dtype=int).tolist())
    sample_frames = []
    frames_processed = 0

    aligned_pils = []

    progress(0, desc="Sampling frames...")
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frames_processed in sample_idxs:
            if source_face is not None:
                frame_faces = fa_app.get(bgr)
                if frame_faces:
                    bgr = swapper.get(bgr, frame_faces[0], source_face, paste_back=True)

            sample_frames.append(bgr)

            aligned_pils.append(
                _digital_defense.get_aligned_face(bgr, input_is_bgr=True)
            )

        frames_processed += 1
        if total > 0:
            progress(frames_processed / total * 0.20, desc="Sampling frames...")
    cap.release()

    progress(0.2, desc="Running digital defenses...")
    digital_scores = _digital_defense(aligned_pils, skip_alignment=True)
    progress(0.6, desc="Running physical defenses...")
    physical_scores = _physical_defense(sample_frames, input_is_bgr=True)

    progress(1.0)

    max_preview = 8
    face_gallery = aligned_pils[:: max(1, len(aligned_pils) // max_preview)][
        :max_preview
    ]

    return (
        {"REAL": digital_scores["deepfake"][0], "FAKE": digital_scores["deepfake"][1]},
        {
            "CLEAN": digital_scores["adversarial"][0],
            "ATTACKED": digital_scores["adversarial"][1],
        },
        {"REAL": physical_scores["spoofed"][0], "FAKE": physical_scores["spoofed"][1]},
        {"CLEAN": physical_scores["patch"][0], "ATTACKED": physical_scores["patch"][1]},
        face_gallery,
    )


with gr.Blocks(title="Computer Vision Spoofing Prevention System with AI") as demo:
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
        deepfake_label = gr.Label(num_top_classes=2, label="Deepfake (LNCLIP-DF)")
        adversarial_label = gr.Label(
            num_top_classes=2, label="Adversarial Attack (AdvGuard)"
        )
        spoofing_label = gr.Label(
            num_top_classes=2, label="Presentation Attack (MiniFAS)"
        )
        patch_label = gr.Label(num_top_classes=2, label="Patch Attack (SAC)")

    face_gallery = gr.Gallery(
        label="Face aligned crops", columns=8, height=160, object_fit="cover"
    )

    deepfake_toggle.change(
        fn=lambda enabled: gr.update(visible=enabled),
        inputs=[deepfake_toggle],
        outputs=[target_image],
    )

    analyze_btn.click(
        fn=process_video,
        inputs=[video_in, attack_toggle, deepfake_toggle, target_image],
        outputs=[
            deepfake_label,
            adversarial_label,
            spoofing_label,
            patch_label,
            face_gallery,
        ],
    )


if __name__ == "__main__":
    demo.launch()
