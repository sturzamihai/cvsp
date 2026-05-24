import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from cvsp.preprocessing import extract_aligned_face_dlib
from cvsp.models.lnclip_df.model import DeepfakeDetectionModel
from cvsp.models.lnclip_df.config import Config as LNClipConfig

import dlib

PREDICTOR_PATH = ROOT / "weights" / "shape_predictor_81_face_landmarks.dat"
LNCLIP_CKPT = ROOT / "weights" / "lnclip.ckpt"
BATCH_SIZE = 32  # frames averaged per verdict
DETECT_WIDTH = 640  # downscale to this width before dlib HOG (speedup on HD+)
MAX_OUT_WIDTH = 1280  # cap annotated-video output width


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


def _scale_for_detection(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a downscaled copy capped at DETECT_WIDTH and the scale factor used."""
    h, w = rgb.shape[:2]
    if w <= DETECT_WIDTH:
        return rgb, 1.0
    scale = DETECT_WIDTH / w
    small = cv2.resize(
        rgb, (DETECT_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA
    )
    return small, scale


def _annotate_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    Draw bounding boxes on an RGB frame.
    Detection runs on a downscaled copy (DETECT_WIDTH) for speed; boxes are
    scaled back to original resolution before drawing.
    """
    small, scale = _scale_for_detection(rgb)
    faces = list(_detector(small, 1))
    out = rgb.copy()

    if not faces:
        cv2.putText(
            out,
            "No face detected",
            (10, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 80, 80),
            2,
        )
        return out

    largest_idx = max(
        range(len(faces)),
        key=lambda i: faces[i].width() * faces[i].height(),
    )
    inv = 1.0 / scale
    for i, face in enumerate(faces):
        x1 = max(0, int(face.left() * inv))
        y1 = max(0, int(face.top() * inv))
        x2 = min(rgb.shape[1] - 1, int(face.right() * inv))
        y2 = min(rgb.shape[0] - 1, int(face.bottom() * inv))
        color = (80, 220, 80) if i == largest_idx else (255, 165, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            "main" if i == largest_idx else "face",
            (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return out


def _get_aligned_pil(rgb: np.ndarray) -> Image.Image | None:
    """
    Run the full DFB alignment pipeline on one RGB frame.
    Downscales to DETECT_WIDTH before detection; output crop is still 256×256.
    Returns a PIL Image (RGB) or None if no face found.
    """
    if _predictor is None:
        return None
    small, _ = _scale_for_detection(rgb)
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    cropped_bgr, _, _ = extract_aligned_face_dlib(_detector, _predictor, bgr)
    if cropped_bgr is None:
        return None
    return Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))


@torch.no_grad()
def _run_lnclip(pil_images: list[Image.Image]) -> torch.Tensor:
    """Returns (N, 2) softmax probabilities [P(REAL), P(FAKE)]."""
    tensors = torch.stack([_lnclip_preprocess(img) for img in pil_images])
    return _lnclip(tensors).logits_labels.softmax(dim=1)


def _reencode_h264(src: str) -> str:
    """Re-encode mp4v → H.264 for browser playback. Falls back if ffmpeg absent."""
    if not shutil.which("ffmpeg"):
        return src
    dst = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    dst.close()
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            src,
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            dst.name,
        ],
        capture_output=True,
    )
    return dst.name if result.returncode == 0 else src


def _to_label(probs: list[float]) -> dict:
    """Format [p_real, p_fake] for gr.Label."""
    return {"REAL": float(probs[0]), "FAKE": float(probs[1])}


def process_video(video_path: str, progress=gr.Progress()):
    if video_path is None:
        return None, None, "No video provided."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, "Could not open video file."

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    # Cap output resolution so the writer stays fast on HD+ webcam clips
    out_scale = min(1.0, MAX_OUT_WIDTH / w)
    out_w, out_h = int(w * out_scale), int(h * out_scale)

    # Sample BATCH_SIZE indices uniformly
    n_samples = min(BATCH_SIZE, max(total, 1))
    sample_idxs = set(np.linspace(0, total - 1, n_samples, dtype=int).tolist())

    raw_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    writer = cv2.VideoWriter(
        raw_tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )

    aligned_pils: list[Image.Image] = []
    frames_processed = 0

    progress(0, desc="Annotating frames…")
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        annotated = _annotate_rgb(rgb)
        out_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        if out_scale < 1.0:
            out_bgr = cv2.resize(out_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(out_bgr)

        if frames_processed in sample_idxs and _lnclip is not None:
            aligned = _get_aligned_pil(rgb)
            if aligned is not None:
                aligned_pils.append(aligned)

        frames_processed += 1
        if total > 0:
            progress(frames_processed / total * 0.80, desc="Annotating frames…")

    cap.release()
    writer.release()

    progress(0.85, desc="Re-encoding for browser…")
    output_path = _reencode_h264(raw_tmp.name)

    label_out = None
    stats_parts = [
        f"Frames: {frames_processed} total, {len(aligned_pils)} sampled with faces.",
    ]

    if not aligned_pils:
        stats_parts.append("No faces detected in sampled frames — classifier skipped.")
    elif _lnclip is None:
        stats_parts.append("LNCLIP-DF model not loaded.")
    else:
        progress(0.93, desc="Running LNCLIP-DF…")
        avg = _run_lnclip(aligned_pils).mean(dim=0).tolist()
        label_out = _to_label(avg)
        verdict = "FAKE" if avg[1] > 0.5 else "REAL"
        stats_parts.append(
            f"Verdict: {verdict}  |  REAL {avg[0]:.1%} / FAKE {avg[1]:.1%}"
        )

    progress(1.0)
    return output_path, label_out, "\n".join(stats_parts)


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

    with gr.Row():
        video_in = gr.Video(
            sources=["webcam", "upload"],
            label="Input — record or upload",
        )
        video_out = gr.Video(label="Annotated output", interactive=False)

    analyze_btn = gr.Button("Analyze", variant="primary")

    with gr.Row():
        verdict_label = gr.Label(num_top_classes=2, label="LNCLIP-DF verdict")
        verdict_stats = gr.Textbox(label="Details", interactive=False, lines=3)

    analyze_btn.click(
        fn=process_video,
        inputs=[video_in],
        outputs=[video_out, verdict_label, verdict_stats],
    )


if __name__ == "__main__":
    demo.launch()
