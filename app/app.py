import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torchattacks
from PIL import Image
from deepface import DeepFace

from cvsp.physical import PhysicalDefense
from cvsp.digital import DigitalDefense
from attacks.deepfake import FaceSwap
from attacks.adversarial_deepfake import AdversarialFGSM

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


PREDICTOR_PATH = ROOT / "weights" / "shape_predictor_81_face_landmarks.dat"
LNCLIP_CKPT = ROOT / "weights" / "lnclip.ckpt"
ADV_GUARD_CKPT = ROOT / "weights" / "adv_guard.pt"
SAC_CKPT = ROOT / "weights" / "apricot_mask.pth"
INSWAPPER_PATH = ROOT / "weights" / "inswapper_128.onnx"
BATCH_SIZE = 32

_device = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

_digital_defense = DigitalDefense(PREDICTOR_PATH, ADV_GUARD_CKPT, LNCLIP_CKPT, _device)
_physical_defense = PhysicalDefense(SAC_CKPT, _device)
_deepfake = FaceSwap(INSWAPPER_PATH)
_adversarial_attack = AdversarialFGSM(_digital_defense.lnclip, _device)


def image_diff(orig, adv):
    orig_arr = np.array(orig).astype(np.float32)
    adv_arr = np.array(adv).astype(np.float32)

    diff = adv_arr - orig_arr
    EPS_U8 = AdversarialFGSM.EPS * 255 * 3
    visible = np.clip((diff / EPS_U8) * 127.5 + 128, 0, 255).astype(np.uint8)

    return Image.fromarray(visible)


def process_video(
    video_path: str,
    apply_attack: bool,
    apply_deepfake: bool,
    target_image: np.ndarray,
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

    if apply_deepfake and target_image is not None:
        _deepfake.prepare(target_image)

    n_samples = min(BATCH_SIZE, max(total, 1))
    sample_idxs = set(np.linspace(0, total - 1, n_samples, dtype=int).tolist())

    sample_frames = []
    deepfake_frames = []

    verified = []
    frames_processed = 0

    aligned_pils = []

    progress(0, desc="Sampling frames...")
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frames_processed in sample_idxs:
            sample_frames.append(bgr)

            if apply_deepfake and _deepfake.ready:
                bgr = _deepfake.apply(bgr)
                deepfake_frames.append(bgr)

            aligned_face = _digital_defense.get_aligned_face(bgr, input_is_bgr=True)

            if aligned_face:
                aligned_pils.append(aligned_face)

            result = DeepFace.verify(target_image, bgr, enforce_detection=False)
            if result:
                verified.append(result["verified"])

        frames_processed += 1
        if total > 0:
            progress(frames_processed / total * 0.20, desc="Sampling frames...")
    cap.release()

    deepfake_slider = gr.update(visible=False)
    if apply_deepfake and deepfake_frames:
        deepfake_slider = gr.update(
            visible=True,
            value=(
                cv2.cvtColor(sample_frames[0], cv2.COLOR_BGR2RGB),
                cv2.cvtColor(deepfake_frames[0], cv2.COLOR_BGR2RGB),
            ),
        )

    original_pils = aligned_pils.copy()
    if apply_attack and aligned_pils:
        progress(0.2, desc="Applying adversarial attack (FGSM)...")
        aligned_pils = _adversarial_attack.apply(aligned_pils)

    adversarial_slider = gr.update(visible=False)
    if apply_attack and aligned_pils:
        adversarial_slider = gr.update(
            visible=True,
            value=(
                aligned_pils[0],
                image_diff(original_pils[0].resize((224, 224)), aligned_pils[0]),
            ),
        )

    progress(0.2, desc="Running digital defenses...")
    digital_scores = _digital_defense(aligned_pils, skip_alignment=True)
    progress(0.6, desc="Running physical defenses...")
    physical_scores = _physical_defense(sample_frames, input_is_bgr=True)

    progress(1.0)

    match_rate = sum(verified) / len(verified) if verified else 0.0

    return (
        {"REAL": digital_scores["deepfake"][0], "FAKE": digital_scores["deepfake"][1]},
        {
            "CLEAN": digital_scores["adversarial"][0],
            "ATTACKED": digital_scores["adversarial"][1],
        },
        {"REAL": physical_scores["spoofed"][0], "FAKE": physical_scores["spoofed"][1]},
        {"CLEAN": physical_scores["patch"][0], "ATTACKED": physical_scores["patch"][1]},
        {"MATCH": match_rate, "NO MATCH": 1.0 - match_rate},
        deepfake_slider,
        adversarial_slider,
    )


_FACE_VERIFY_CSS = """
.face-verify-box {
    border: 2px solid var(--primary-500) !important;
    border-radius: 8px;
    background: var(--primary-50);
}
"""

with gr.Blocks(
    title="Computer Vision Spoofing Prevention System with AI", css=_FACE_VERIFY_CSS
) as demo:
    gr.Markdown("# Computer Vision Spoofing Prevention System with AI\n")

    with gr.Row(equal_height=True):
        target_image = gr.Image(
            label="Target face",
            type="numpy",
        )
        with gr.Column():
            video_in = gr.Video(
                sources=["webcam", "upload"],
                label="Input — record or upload",
            )
            attack_toggle = gr.Checkbox(
                label="Apply adversarial attack",
                value=False,
                info="Applies FGSM perturbation to sampled frames to evade the deepfake detector.",
            )
            deepfake_toggle = gr.Checkbox(
                label="Apply deepfake",
                value=False,
                info="Swaps faces in the video with the target face using inswapper.",
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

    with gr.Row():
        face_verify_label = gr.Label(
            num_top_classes=2,
            label="Face Verification (DeepFace)",
            elem_classes=["face-verify-box"],
        )

    with gr.Row():
        deepfake_slider = gr.ImageSlider(visible=False)

    with gr.Row():
        adversarial_slider = gr.ImageSlider(visible=False)

    analyze_btn.click(
        fn=process_video,
        inputs=[video_in, attack_toggle, deepfake_toggle, target_image],
        outputs=[
            deepfake_label,
            adversarial_label,
            spoofing_label,
            patch_label,
            face_verify_label,
            deepfake_slider,
            adversarial_slider,
        ],
    )


if __name__ == "__main__":
    demo.launch()
