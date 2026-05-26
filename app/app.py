import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torchattacks
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from cvsp.preprocessing import extract_aligned_face_dlib
from cvsp.models.lnclip_df.model import DeepfakeDetectionModel
from cvsp.models.lnclip_df.config import Config as LNClipConfig
from cvsp.models.minifas import FaceAntiSpoofing

import dlib

PREDICTOR_PATH = ROOT / "weights" / "shape_predictor_81_face_landmarks.dat"
LNCLIP_CKPT = ROOT / "weights" / "lnclip.ckpt"
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


def _load_lnclip():
    if not LNCLIP_CKPT.exists():
        print(f"[LNCLIP] Checkpoint not found at {LNCLIP_CKPT}.")
        return None, None
    try:
        ckpt = torch.load(str(LNCLIP_CKPT), map_location="cpu")
        model = DeepfakeDetectionModel(LNClipConfig(**ckpt["hyper_parameters"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        preprocess = model.get_preprocessing()
        print("[LNCLIP] Model loaded successfully.")
        return model, preprocess
    except Exception as exc:
        print(f"[LNCLIP] Could not load model: {exc}")
        return None, None


_detector, _predictor = _load_dlib()
_lnclip, _lnclip_preprocess = _load_lnclip()
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


def _get_aligned_pil(rgb: np.ndarray) -> Image.Image | None:
    """
    Run the full DFB alignment pipeline on one RGB frame.
    Alignment runs on the original full-resolution frame so the 256×256 crop
    matches the quality of the offline training preprocessing.
    Returns a PIL Image (RGB) or None if no face found.
    """
    if _predictor is None:
        return None
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cropped_bgr, _, _ = extract_aligned_face_dlib(_detector, _predictor, bgr)
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
    """Format [p_real, p_fake] for gr.Label."""
    return {"REAL": float(probs[0]), "FAKE": float(probs[1])}


def process_video(video_path: str, apply_attack: bool = False, progress=gr.Progress()):
    if video_path is None:
        return None, None, "No video provided."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, "Could not open video file."

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
            if _lnclip is not None:
                aligned = _get_aligned_pil(rgb)
                if aligned is not None:
                    aligned_pils.append(aligned)

        frames_processed += 1
        if total > 0:
            progress(frames_processed / total * 0.90, desc="Sampling frames…")

    cap.release()

    label_out = None
    spoof_label_out = None
    stats_parts = [
        f"Frames: {frames_processed} total, {len(aligned_pils)} sampled with faces.",
    ]

    if spoof_results:
        avg_live = sum(score for _, score in spoof_results) / len(spoof_results)
        spoof_label_out = {"LIVE": avg_live, "SPOOF": 1.0 - avg_live}
        verdict_spoof = "SPOOF" if avg_live < 0.5 else "LIVE"
        stats_parts.append(
            f"Anti-spoof: {verdict_spoof}  |  LIVE {avg_live:.1%} / SPOOF {1-avg_live:.1%}"
            f"  ({len(spoof_results)} frames checked)"
        )
    else:
        stats_parts.append("Anti-spoof: no face matched across detectors.")

    if not aligned_pils:
        stats_parts.append("No faces detected in sampled frames — classifier skipped.")
    elif _lnclip is None:
        stats_parts.append("LNCLIP-DF model not loaded.")
    else:
        progress(0.95, desc="Running LNCLIP-DF…")
        avg = _run_lnclip(aligned_pils, apply_attack=apply_attack).mean(dim=0).tolist()
        label_out = _to_label(avg)
        verdict = "FAKE" if avg[1] > 0.5 else "REAL"
        stats_parts.append(
            f"Verdict: {verdict}  |  REAL {avg[0]:.1%} / FAKE {avg[1]:.1%}"
        )

    progress(1.0)
    return label_out, spoof_label_out, "\n".join(stats_parts)


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
        spoof_label = gr.Label(num_top_classes=2, label="Anti-spoof (MiniFAS)")
        verdict_label = gr.Label(num_top_classes=2, label="Deepfake (LNCLIP-DF)")
        verdict_stats = gr.Textbox(label="Details", interactive=False, lines=3)

    analyze_btn.click(
        fn=process_video,
        inputs=[video_in, attack_toggle],
        outputs=[verdict_label, spoof_label, verdict_stats],
    )


if __name__ == "__main__":
    demo.launch()
